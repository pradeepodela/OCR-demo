import streamlit as st
import json
import os
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, create_model
from typing import Dict, Any, List
import base64
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai import Mistral
from mistralai.models import OCRResponse
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Initialize Mistral client
@st.cache_resource
def get_mistral_client():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        st.error("MISTRAL_API_KEY not found in environment variables")
        st.stop()
    return Mistral(api_key=api_key)

client = get_mistral_client()

class DocumentType(Enum):
    PDF = "PDF"
    IMAGE = "IMAGE"

class StructuredOCR(BaseModel):
    file_name: str
    topics: List[str]
    languages: str
    ocr_contents: Dict[str, Any]

def replace_images_in_markdown(markdown_str: str, images_dict: Dict[str, str]) -> str:
    """Replace image placeholders in markdown with base64-encoded images."""
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """Combine OCR text and images into a single markdown document."""
    markdowns: List[str] = []
    
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    
    return "\n\n".join(markdowns)

def create_dynamic_model(schema_dict: Dict[str, Any], model_name: str = "CustomOCRModel") -> BaseModel:
    """Create a dynamic Pydantic model from a JSON schema dictionary."""
    fields = {}
    
    for field_name, field_info in schema_dict.items():
        if isinstance(field_info, dict) and "type" in field_info:
            field_type = field_info["type"]
            default_value = field_info.get("default", ...)
            
            if field_type == "string":
                fields[field_name] = (str, default_value)
            elif field_type == "integer":
                fields[field_name] = (int, default_value)
            elif field_type == "array":
                fields[field_name] = (List[str], default_value)
            elif field_type == "object":
                fields[field_name] = (Dict[str, Any], default_value)
            else:
                fields[field_name] = (str, default_value)
        else:
            fields[field_name] = (str, ...)
    
    return create_model(model_name, **fields)

def process_file_locally(uploaded_file, doc_type: str, ocr_mode: str, custom_model: BaseModel = None):
    """Process uploaded file using local temporary storage."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Process based on document type
        if doc_type == "PDF":
            # For PDF - use file path directly with Mistral API
            result = process_pdf_file(tmp_file_path, ocr_mode, custom_model)
        else:
            # For IMAGE - convert to base64 and use data URL
            result = process_image_file(uploaded_file, ocr_mode, custom_model)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return result
        
    except Exception as e:
        # Clean up on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise e

def process_pdf_file(file_path: str, ocr_mode: str, custom_model: BaseModel = None):
    """Process PDF file using local file path."""
    try:
        # Read file and convert to base64 for API
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        base64_content = base64.b64encode(file_content).decode('utf-8')
        data_url = f"data:application/pdf;base64,{base64_content}"
        
        if ocr_mode == "Simple OCR":
            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=data_url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            combined_markdown = get_combined_markdown(pdf_response)
            return json.dumps({"markdown": combined_markdown}, indent=4)
        
        else:  # Structured OCR
            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=data_url),
                model="mistral-ocr-latest"
            )
            pdf_ocr_markdown = "\n\n".join(page.markdown for page in pdf_response.pages)
            response_format = custom_model if custom_model else StructuredOCR
            
            chat_response = client.chat.parse(
                model="pixtral-12b-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            TextChunk(text=(
                                f"This is the PDF's OCR in markdown:\n{pdf_ocr_markdown}\n.\n"
                                "Convert this into a structured JSON response "
                                "with the OCR contents in a sensible dictionary."
                            ))
                        ]
                    }
                ],
                response_format=response_format,
                temperature=0
            )
            
            result = chat_response.choices[0].message.parsed
            return result.model_dump_json(indent=4)
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return json.dumps({"error": str(e)}, indent=4)

def process_image_file(uploaded_file, ocr_mode: str, custom_model: BaseModel = None):
    """Process image file using base64 data URL."""
    try:
        # Convert to base64 data URL
        file_bytes = uploaded_file.getvalue()
        base64_encoded = base64.b64encode(file_bytes).decode('utf-8')
        
        file_extension = Path(uploaded_file.name).suffix.lower()
        mime_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        
        mime_type = mime_type_map.get(file_extension, 'image/jpeg')
        data_url = f"data:{mime_type};base64,{base64_encoded}"
        
        if ocr_mode == "Simple OCR":
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=data_url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            combined_markdown = get_combined_markdown(image_response)
            return json.dumps({"markdown": combined_markdown}, indent=4)
        
        else:  # Structured OCR
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=data_url),
                model="mistral-ocr-latest"
            )
            image_ocr_markdown = image_response.pages[0].markdown
            
            response_format = custom_model if custom_model else StructuredOCR
            
            chat_response = client.chat.parse(
                model="pixtral-12b-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            ImageURLChunk(image_url=data_url),
                            TextChunk(text=(
                                f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n.\n"
                                "Convert this into a structured JSON response "
                                "with the OCR contents in a sensible dictionary."
                            ))
                        ]
                    }
                ],
                response_format=response_format,
                temperature=0
            )
            
            result = chat_response.choices[0].message.parsed
            return result.model_dump_json(indent=4)
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return json.dumps({"error": str(e)}, indent=4)

def process_url(url: str, doc_type: str, ocr_mode: str, custom_model: BaseModel = None):
    """Process document from URL."""
    try:
        if ocr_mode == "Simple OCR":
            if doc_type == "PDF":
                pdf_response = client.ocr.process(
                    document=DocumentURLChunk(document_url=url),
                    model="mistral-ocr-latest",
                    include_image_base64=True
                )
                combined_markdown = get_combined_markdown(pdf_response)
                return json.dumps({"markdown": combined_markdown}, indent=4)
            else:
                image_response = client.ocr.process(
                    document=ImageURLChunk(image_url=url),
                    model="mistral-ocr-latest",
                    include_image_base64=True
                )
                combined_markdown = get_combined_markdown(image_response)
                return json.dumps({"markdown": combined_markdown}, indent=4)
        
        else:  # Structured OCR
            if doc_type == "PDF":
                pdf_response = client.ocr.process(
                    document=DocumentURLChunk(document_url=url),
                    model="mistral-ocr-latest"
                )
                ocr_markdown = "\n\n".join(page.markdown for page in pdf_response.pages)
                content = [TextChunk(text=(
                    f"This is the PDF's OCR in markdown:\n{ocr_markdown}\n.\n"
                    "Convert this into a structured JSON response "
                    "with the OCR contents in a sensible dictionary."
                ))]
            else:
                image_response = client.ocr.process(
                    document=ImageURLChunk(image_url=url),
                    model="mistral-ocr-latest"
                )
                ocr_markdown = image_response.pages[0].markdown
                content = [
                    ImageURLChunk(image_url=url),
                    TextChunk(text=(
                        f"This is the image's OCR in markdown:\n{ocr_markdown}\n.\n"
                        "Convert this into a structured JSON response "
                        "with the OCR contents in a sensible dictionary."
                    ))
                ]
            
            response_format = custom_model if custom_model else StructuredOCR
            
            chat_response = client.chat.parse(
                model="pixtral-12b-latest",
                messages=[{"role": "user", "content": content}],
                response_format=response_format,
                temperature=0
            )
            
            result = chat_response.choices[0].message.parsed
            return result.model_dump_json(indent=4)
            
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")
        return json.dumps({"error": str(e)}, indent=4)

def main():
    st.set_page_config(
        page_title="OCR Application",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 OCR Application")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        ocr_mode = st.selectbox(
            "OCR Mode",
            ["Simple OCR", "Structured OCR"]
        )
        
        doc_type = st.selectbox(
            "Document Type",
            ["IMAGE", "PDF"]
        )
        
        # Custom schema for structured OCR
        custom_model = None
        if ocr_mode == "Structured OCR":
            st.subheader("Custom Schema")
            use_custom = st.checkbox("Use Custom Schema")
            
            if use_custom:
                schema_text = st.text_area(
                    "JSON Schema",
                    height=150,
                    placeholder='{\n  "field_name": {"type": "string"},\n  "amount": {"type": "string"}\n}'
                )
                
                if schema_text:
                    try:
                        schema_dict = json.loads(schema_text)
                        custom_model = create_dynamic_model(schema_dict)
                        st.success("Custom schema loaded!")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON schema")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Input")
        
        input_method = "Upload File"
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose file",
                type=['png', 'jpg', 'jpeg', 'pdf']
            )
            
            if uploaded_file:
                st.success(f"File: {uploaded_file.name}")
                
                if uploaded_file.type.startswith('image/'):
                    st.image(uploaded_file, caption="Preview", use_container_width=True)
                
                if st.button("🚀 Process", type="primary"):
                    with st.spinner("Processing..."):
                        result = process_file_locally(uploaded_file, doc_type, ocr_mode, custom_model)
                        st.session_state.result = result
                        st.rerun()
        
        else:
            url = st.text_input("Document URL")
            
            if url and st.button("🚀 Process", type="primary"):
                with st.spinner("Processing..."):
                    result = process_url(url, doc_type, ocr_mode, custom_model)
                    st.session_state.result = result
                    st.rerun()
    
    with col2:
        st.subheader("📊 Results")
        
        if 'result' in st.session_state:
            try:
                data = json.loads(st.session_state.result)
                
                if "markdown" in data:
                    st.markdown("**Extracted Content:**")
                    st.markdown(data["markdown"])
                else:
                    st.json(data)
                
                st.download_button(
                    "💾 Download Results",
                    data=st.session_state.result,
                    file_name="ocr_results.json",
                    mime="application/json"
                )
                
            except json.JSONDecodeError:
                st.text(st.session_state.result)
        else:
            st.info("Process a document to see results here")

if __name__ == "__main__":
    main()
