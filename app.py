# import streamlit as st
# import json
# import os
# from enum import Enum
# from pathlib import Path
# from pydantic import BaseModel, create_model
# from typing import Dict, Any, List
# import base64
# from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
# from mistralai import Mistral
# from mistralai.models import OCRResponse
# from dotenv import load_dotenv
# import tempfile

# # Load environment variables
# load_dotenv()

# # Initialize Mistral client
# @st.cache_resource
# def get_mistral_client():
#     api_key = os.environ.get("MISTRAL_API_KEY")
#     if not api_key:
#         st.error("MISTRAL_API_KEY not found in environment variables")
#         st.stop()
#     return Mistral(api_key=api_key)

# client = get_mistral_client()

# class DocumentType(Enum):
#     PDF = "PDF"
#     IMAGE = "IMAGE"

# class StructuredOCR(BaseModel):
#     file_name: str
#     topics: List[str]
#     languages: str
#     ocr_contents: Dict[str, Any]

# def replace_images_in_markdown(markdown_str: str, images_dict: Dict[str, str]) -> str:
#     """Replace image placeholders in markdown with base64-encoded images."""
#     for img_name, base64_str in images_dict.items():
#         markdown_str = markdown_str.replace(
#             f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
#         )
#     return markdown_str

# def get_combined_markdown(ocr_response: OCRResponse) -> str:
#     """Combine OCR text and images into a single markdown document."""
#     markdowns: List[str] = []
    
#     for page in ocr_response.pages:
#         image_data = {}
#         for img in page.images:
#             image_data[img.id] = img.image_base64
        
#         markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    
#     return "\n\n".join(markdowns)

# def create_dynamic_model(schema_dict: Dict[str, Any], model_name: str = "CustomOCRModel") -> BaseModel:
#     """Create a dynamic Pydantic model from a JSON schema dictionary."""
#     fields = {}
    
#     for field_name, field_info in schema_dict.items():
#         if isinstance(field_info, dict) and "type" in field_info:
#             field_type = field_info["type"]
#             default_value = field_info.get("default", ...)
            
#             if field_type == "string":
#                 fields[field_name] = (str, default_value)
#             elif field_type == "integer":
#                 fields[field_name] = (int, default_value)
#             elif field_type == "array":
#                 fields[field_name] = (List[str], default_value)
#             elif field_type == "object":
#                 fields[field_name] = (Dict[str, Any], default_value)
#             else:
#                 fields[field_name] = (str, default_value)
#         else:
#             fields[field_name] = (str, ...)
    
#     return create_model(model_name, **fields)

# def process_file_locally(uploaded_file, doc_type: str, ocr_mode: str, custom_model: BaseModel = None):
#     """Process uploaded file using local temporary storage."""
#     try:
#         # Create temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             tmp_file_path = tmp_file.name
        
#         # Process based on document type
#         if doc_type == "PDF":
#             # For PDF - use file path directly with Mistral API
#             result = process_pdf_file(tmp_file_path, ocr_mode, custom_model)
#         else:
#             # For IMAGE - convert to base64 and use data URL
#             result = process_image_file(uploaded_file, ocr_mode, custom_model)
        
#         # Clean up temporary file
#         os.unlink(tmp_file_path)
        
#         return result
        
#     except Exception as e:
#         # Clean up on error
#         if 'tmp_file_path' in locals():
#             try:
#                 os.unlink(tmp_file_path)
#             except:
#                 pass
#         raise e

# def process_pdf_file(file_path: str, ocr_mode: str, custom_model: BaseModel = None):
#     """Process PDF file using local file path."""
#     try:
#         # Read file and convert to base64 for API
#         with open(file_path, 'rb') as f:
#             file_content = f.read()
        
#         base64_content = base64.b64encode(file_content).decode('utf-8')
#         data_url = f"data:application/pdf;base64,{base64_content}"
        
#         if ocr_mode == "Simple OCR":
#             pdf_response = client.ocr.process(
#                 document=DocumentURLChunk(document_url=data_url),
#                 model="mistral-ocr-latest",
#                 include_image_base64=True
#             )
#             combined_markdown = get_combined_markdown(pdf_response)
#             return json.dumps({"markdown": combined_markdown}, indent=4)
        
#         else:  # Structured OCR
#             pdf_response = client.ocr.process(
#                 document=DocumentURLChunk(document_url=data_url),
#                 model="mistral-ocr-latest"
#             )
#             pdf_ocr_markdown = "\n\n".join(page.markdown for page in pdf_response.pages)
#             response_format = custom_model if custom_model else StructuredOCR
            
#             chat_response = client.chat.parse(
#                 model="pixtral-12b-latest",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             TextChunk(text=(
#                                 f"This is the PDF's OCR in markdown:\n{pdf_ocr_markdown}\n.\n"
#                                 "Convert this into a structured JSON response "
#                                 "with the OCR contents in a sensible dictionary."
#                             ))
#                         ]
#                     }
#                 ],
#                 response_format=response_format,
#                 temperature=0
#             )
            
#             result = chat_response.choices[0].message.parsed
#             return result.model_dump_json(indent=4)
            
#     except Exception as e:
#         st.error(f"Error processing PDF: {str(e)}")
#         return json.dumps({"error": str(e)}, indent=4)

# def process_image_file(uploaded_file, ocr_mode: str, custom_model: BaseModel = None):
#     """Process image file using base64 data URL."""
#     try:
#         # Convert to base64 data URL
#         file_bytes = uploaded_file.getvalue()
#         base64_encoded = base64.b64encode(file_bytes).decode('utf-8')
        
#         file_extension = Path(uploaded_file.name).suffix.lower()
#         mime_type_map = {
#             '.png': 'image/png',
#             '.jpg': 'image/jpeg',
#             '.jpeg': 'image/jpeg'
#         }
        
#         mime_type = mime_type_map.get(file_extension, 'image/jpeg')
#         data_url = f"data:{mime_type};base64,{base64_encoded}"
        
#         if ocr_mode == "Simple OCR":
#             image_response = client.ocr.process(
#                 document=ImageURLChunk(image_url=data_url),
#                 model="mistral-ocr-latest",
#                 include_image_base64=True
#             )
#             combined_markdown = get_combined_markdown(image_response)
#             return json.dumps({"markdown": combined_markdown}, indent=4)
        
#         else:  # Structured OCR
#             image_response = client.ocr.process(
#                 document=ImageURLChunk(image_url=data_url),
#                 model="mistral-ocr-latest"
#             )
#             image_ocr_markdown = image_response.pages[0].markdown
            
#             response_format = custom_model if custom_model else StructuredOCR
            
#             chat_response = client.chat.parse(
#                 model="pixtral-12b-latest",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             ImageURLChunk(image_url=data_url),
#                             TextChunk(text=(
#                                 f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n.\n"
#                                 "Convert this into a structured JSON response "
#                                 "with the OCR contents in a sensible dictionary."
#                             ))
#                         ]
#                     }
#                 ],
#                 response_format=response_format,
#                 temperature=0
#             )
            
#             result = chat_response.choices[0].message.parsed
#             return result.model_dump_json(indent=4)
            
#     except Exception as e:
#         st.error(f"Error processing image: {str(e)}")
#         return json.dumps({"error": str(e)}, indent=4)

# def process_url(url: str, doc_type: str, ocr_mode: str, custom_model: BaseModel = None):
#     """Process document from URL."""
#     try:
#         if ocr_mode == "Simple OCR":
#             if doc_type == "PDF":
#                 pdf_response = client.ocr.process(
#                     document=DocumentURLChunk(document_url=url),
#                     model="mistral-ocr-latest",
#                     include_image_base64=True
#                 )
#                 combined_markdown = get_combined_markdown(pdf_response)
#                 return json.dumps({"markdown": combined_markdown}, indent=4)
#             else:
#                 image_response = client.ocr.process(
#                     document=ImageURLChunk(image_url=url),
#                     model="mistral-ocr-latest",
#                     include_image_base64=True
#                 )
#                 combined_markdown = get_combined_markdown(image_response)
#                 return json.dumps({"markdown": combined_markdown}, indent=4)
        
#         else:  # Structured OCR
#             if doc_type == "PDF":
#                 pdf_response = client.ocr.process(
#                     document=DocumentURLChunk(document_url=url),
#                     model="mistral-ocr-latest"
#                 )
#                 ocr_markdown = "\n\n".join(page.markdown for page in pdf_response.pages)
#                 content = [TextChunk(text=(
#                     f"This is the PDF's OCR in markdown:\n{ocr_markdown}\n.\n"
#                     "Convert this into a structured JSON response "
#                     "with the OCR contents in a sensible dictionary."
#                 ))]
#             else:
#                 image_response = client.ocr.process(
#                     document=ImageURLChunk(image_url=url),
#                     model="mistral-ocr-latest"
#                 )
#                 ocr_markdown = image_response.pages[0].markdown
#                 content = [
#                     ImageURLChunk(image_url=url),
#                     TextChunk(text=(
#                         f"This is the image's OCR in markdown:\n{ocr_markdown}\n.\n"
#                         "Convert this into a structured JSON response "
#                         "with the OCR contents in a sensible dictionary."
#                     ))
#                 ]
            
#             response_format = custom_model if custom_model else StructuredOCR
            
#             chat_response = client.chat.parse(
#                 model="pixtral-12b-latest",
#                 messages=[{"role": "user", "content": content}],
#                 response_format=response_format,
#                 temperature=0
#             )
            
#             result = chat_response.choices[0].message.parsed
#             return result.model_dump_json(indent=4)
            
#     except Exception as e:
#         st.error(f"Error processing URL: {str(e)}")
#         return json.dumps({"error": str(e)}, indent=4)

# def main():
#     st.set_page_config(
#         page_title="OCR Application",
#         page_icon="📄",
#         layout="wide"
#     )
    
#     st.title("📄 OCR Application")
#     st.markdown("---")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Settings")
        
#         ocr_mode = st.selectbox(
#             "OCR Mode",
#             ["Simple OCR", "Structured OCR"]
#         )
        
#         doc_type = st.selectbox(
#             "Document Type",
#             ["IMAGE", "PDF"]
#         )
        
#         # Custom schema for structured OCR
#         custom_model = None
#         if ocr_mode == "Structured OCR":
#             st.subheader("Custom Schema")
#             use_custom = st.checkbox("Use Custom Schema")
            
#             if use_custom:
#                 schema_text = st.text_area(
#                     "JSON Schema",
#                     height=150,
#                     placeholder='{\n  "field_name": {"type": "string"},\n  "amount": {"type": "string"}\n}'
#                 )
                
#                 if schema_text:
#                     try:
#                         schema_dict = json.loads(schema_text)
#                         custom_model = create_dynamic_model(schema_dict)
#                         st.success("Custom schema loaded!")
#                     except json.JSONDecodeError:
#                         st.error("Invalid JSON schema")
    
#     # Main content
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.subheader("📤 Input")
        
#         input_method = "Upload File"
        
#         if input_method == "Upload File":
#             uploaded_file = st.file_uploader(
#                 "Choose file",
#                 type=['png', 'jpg', 'jpeg', 'pdf']
#             )
            
#             if uploaded_file:
#                 st.success(f"File: {uploaded_file.name}")
                
#                 if uploaded_file.type.startswith('image/'):
#                     st.image(uploaded_file, caption="Preview", use_container_width=True)
                
#                 if st.button("🚀 Process", type="primary"):
#                     with st.spinner("Processing..."):
#                         result = process_file_locally(uploaded_file, doc_type, ocr_mode, custom_model)
#                         st.session_state.result = result
#                         st.rerun()
        
#         else:
#             url = st.text_input("Document URL")
            
#             if url and st.button("🚀 Process", type="primary"):
#                 with st.spinner("Processing..."):
#                     result = process_url(url, doc_type, ocr_mode, custom_model)
#                     st.session_state.result = result
#                     st.rerun()
    
#     with col2:
#         st.subheader("📊 Results")
        
#         if 'result' in st.session_state:
#             try:
#                 data = json.loads(st.session_state.result)
                
#                 if "markdown" in data:
#                     st.markdown("**Extracted Content:**")
#                     st.markdown(data["markdown"])
#                 else:
#                     st.json(data)
                
#                 st.download_button(
#                     "💾 Download Results",
#                     data=st.session_state.result,
#                     file_name="ocr_results.json",
#                     mime="application/json"
#                 )
                
#             except json.JSONDecodeError:
#                 st.text(st.session_state.result)
#         else:
#             st.info("Process a document to see results here")

# if __name__ == "__main__":
#     main()







# app.py
import base64, json, os, tempfile, mimetypes
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
import re
import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from pydantic import BaseModel, create_model, ValidationError
from openai import OpenAI                           # OpenRouter compatible
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaValidationError

# ------------------------------------------------------------------ #
# 0.  ENV + CLIENTS
# ------------------------------------------------------------------ #
load_dotenv()

@st.cache_resource(show_spinner=False)
def get_mistral() -> Mistral:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        st.error("❌  Set MISTRAL_API_KEY in .env")
        st.stop()
    return Mistral(api_key=key)

@st.cache_resource(show_spinner=False)
def get_openrouter() -> OpenAI | None:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

mistral_client = get_mistral()
openrouter_client = get_openrouter()

# ------------------------------------------------------------------ #
# 1.  JSON‑SCHEMA ➜  Pydantic builder
# ------------------------------------------------------------------ #
_json2py = {
    "string": str, "integer": int, "number": float,
    "boolean": bool, "null": type(None), "object": Dict[str, Any]
}
def _safe(name: str) -> str:                       # valid python identifier
    out = name.strip().replace(" ", "_").replace("-", "_")
    return out if out.isidentifier() else f"field_{abs(hash(out))}"

def _node2type(node: Dict[str, Any]) -> Any:
    t = node.get("type", "string")
    if t in _json2py: return _json2py[t]
    if t == "array": return List[_node2type(node.get("items", {"type":"string"}))]  # type: ignore
    if t == "object":
        props = node.get("properties", {})
        sub = { _safe(k): (_node2type(v), ...) for k, v in props.items() }
        return create_model("Nested", **sub)                                         # type: ignore
    return Any

def dynamic_model(schema: Dict[str, Any], *, name="CustomModel") -> BaseModel:
    props = schema.get("properties", schema)
    fields = { _safe(k): (_node2type(v), v.get("default", ...))
               for k, v in props.items() }
    return create_model(name, **fields)                                              # type: ignore

# ------------------------------------------------------------------ #
# 2.  OCR helpers
# ------------------------------------------------------------------ #
def _merge_md(resp: OCRResponse) -> str:
    md_pages = []
    for p in resp.pages:
        imgs = {i.id: i.image_base64 for i in p.images}
        md = p.markdown
        for iid, b64 in imgs.items():
            md = md.replace(f"![{iid}]({iid})", f"![{iid}]({b64})")
        md_pages.append(md)
    return "\n\n".join(md_pages)

# ------------------------------------------------------------------ #
# 3.  JSON Schema Validation & Comparison
# ------------------------------------------------------------------ #
def validate_json_against_schema(json_data: dict, schema: dict) -> tuple[bool, list]:
    """
    Validate JSON data against a schema and return validation results.
    Returns (is_valid, errors_list)
    """
    try:
        validate(instance=json_data, schema=schema)
        return True, []
    except JsonSchemaValidationError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Validation error: {str(e)}"]

def compare_json_structures(json1: dict, json2: dict, path: str = "") -> list:
    """
    Compare two JSON structures and return differences.
    Returns list of difference descriptions.
    """
    differences = []
    
    # Get all keys from both JSONs
    all_keys = set(json1.keys()) | set(json2.keys())
    
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        
        if key not in json1:
            differences.append(f"Missing in JSON1: {current_path}")
        elif key not in json2:
            differences.append(f"Missing in JSON2: {current_path}")
        else:
            val1, val2 = json1[key], json2[key]
            
            # Check if both are dicts
            if isinstance(val1, dict) and isinstance(val2, dict):
                differences.extend(compare_json_structures(val1, val2, current_path))
            # Check if both are lists
            elif isinstance(val1, list) and isinstance(val2, list):
                if len(val1) != len(val2):
                    differences.append(f"Array length differs at {current_path}: {len(val1)} vs {len(val2)}")
                else:
                    for i, (item1, item2) in enumerate(zip(val1, val2)):
                        if isinstance(item1, dict) and isinstance(item2, dict):
                            differences.extend(compare_json_structures(item1, item2, f"{current_path}[{i}]"))
                        elif item1 != item2:
                            differences.append(f"Array item differs at {current_path}[{i}]: {item1} vs {item2}")
            # Check if values are different
            elif val1 != val2:
                differences.append(f"Value differs at {current_path}: {val1} vs {val2}")
    
    return differences

def calculate_schema_compliance_score(json_data: dict, schema: dict) -> tuple[float, dict]:
    """
    Calculate a compliance score based on how well the JSON matches the schema.
    Returns (score, details)
    """
    details = {
        "required_fields_present": 0,
        "total_required_fields": 0,
        "optional_fields_present": 0,
        "total_optional_fields": 0,
        "type_matches": 0,
        "total_type_checks": 0,
        "validation_errors": []
    }
    
    # Check validation first
    is_valid, errors = validate_json_against_schema(json_data, schema)
    details["validation_errors"] = errors
    
    if not is_valid:
        details["overall_valid"] = False
    else:
        details["overall_valid"] = True
    
    # Count required fields
    required_fields = schema.get("required", [])
    details["total_required_fields"] = len(required_fields)
    
    for field in required_fields:
        if field in json_data:
            details["required_fields_present"] += 1
    
    # Count optional fields
    all_properties = schema.get("properties", {})
    optional_fields = [k for k in all_properties.keys() if k not in required_fields]
    details["total_optional_fields"] = len(optional_fields)
    
    for field in optional_fields:
        if field in json_data:
            details["optional_fields_present"] += 1
    
    # Calculate overall score
    required_score = details["required_fields_present"] / max(details["total_required_fields"], 1)
    optional_score = details["optional_fields_present"] / max(details["total_optional_fields"], 1)
    validation_score = 1.0 if details["overall_valid"] else 0.5
    
    overall_score = (required_score * 0.6 + optional_score * 0.2 + validation_score * 0.2) * 100
    
    return overall_score, details

# ------------------------------------------------------------------ #
# 4.  LLM wrappers  (temperature = 0 everywhere)
# ------------------------------------------------------------------ #
DEFAULT_MODEL = "pixtral-12b-latest"
OPENROUTER_MODEL = "qwen/qwen2.5-vl-72b-instruct"

class FallbackSchema(BaseModel):
    file_name: str
    ocr_contents: Dict[str, Any]

def _create_extraction_prompt(ocr_text: str, schema: str) -> str:
    """Create a focused prompt for JSON extraction from OCR text."""
    return f"""
    
    You are an expert data extraction specialist. Your task is to extract information from the OCR text below and convert it into JSON format according to the provided schema. Follow these instructions carefully:

### **INSTRUCTIONS:**
1. **Read the OCR Text Carefully**: Identify all relevant information from the OCR text.
2. **Extract Data According to the Schema**: Match the extracted information to the fields in the JSON schema.
3. **Use Exact Values**: Extract values directly from the document without modifying or interpreting them.
4. **Monetary Amounts**: For monetary amounts, extract only the numeric value (remove currency symbols).
5. **Dates**: Format dates as `YYYY-MM-DD`.
6. **Handle Missing Information**:
   - For **optional fields**, use `null` if the information is not present in the document.
   - For **required fields**, make reasonable inferences if the information is unclear or missing.
7. **Return ONLY Valid JSON**: Ensure the output is a valid JSON object with no additional text or explanations.

### **OCR TEXT:**
{ocr_text}

### **JSON SCHEMA:**
{schema}

### **Your Task**:
- Extract all relevant information from the OCR text.
- Convert the extracted data into JSON format based on the provided schema.
- Ensure the JSON output adheres to the schema's structure and data types.
- Return ONLY the valid JSON output.


"""

def _mistral_parse(chunks, model_cls):
    chat = mistral_client.chat.parse(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": chunks}],
        response_format=model_cls,
        temperature=0,  # Set to 0 for maximum consistency
    )
    return json.loads(chat.choices[0].message.parsed.model_dump_json())

def _extract_json(text: str) -> dict:
    """Extract the first JSON block from text."""
    try:
        # Try parsing directly
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract JSON from text block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError("Failed to extract valid JSON from OpenRouter response.")

def _openrouter_parse(prompt_text, img_url, model_cls):  # OpenRouter fallback
    if not openrouter_client:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    content = []
    if img_url:
        content.append({"type": "image_url", "image_url": {"url": img_url}})
    content.append({"type": "text", "text": prompt_text})

    completion = openrouter_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0,  # Set to 0 for maximum consistency
        extra_headers={
            "X-Title": "StreamlitOCR",
            "HTTP-Referer": "https://localhost"
        },
    )
    raw = completion.choices[0].message.content.strip()
    
    return _extract_json(raw)

# ------------------------------------------------------------------ #
# 5.  Processing: PDF / Image  ->  OCR markdown
# ------------------------------------------------------------------ #
def ocr_pdf(path: str, simple: bool) -> tuple[str, str]:
    """return merged_markdown, data_url"""
    with open(path, "rb") as f: data = f.read()
    b64 = base64.b64encode(data).decode()
    url = f"data:application/pdf;base64,{b64}"

    resp = mistral_client.ocr.process(
        document=DocumentURLChunk(document_url=url),
        model="mistral-ocr-latest",
        include_image_base64=True if simple else False,
    )
    return (_merge_md(resp), url)

def ocr_image(uploaded, simple: bool) -> tuple[str, str]:
    data = uploaded.getvalue()
    b64 = base64.b64encode(data).decode()
    mime = mimetypes.guess_type(uploaded.name)[0] or "image/jpeg"
    url = f"data:{mime};base64,{b64}"

    resp = mistral_client.ocr.process(
        document=ImageURLChunk(image_url=url),
        model="mistral-ocr-latest",
        include_image_base64=True if simple else False,
    )
    return (_merge_md(resp), url)

# ------------------------------------------------------------------ #
# 6.  Streamlit UI
# ------------------------------------------------------------------ #
class DocType(Enum): IMAGE="IMAGE"; PDF="PDF"

st.set_page_config("OCR Comparator", "📄", layout="wide")
st.title("📄 OCR → JSON Comparator")
st.markdown("---")

# ---------------- Sidebar ---------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    ocr_mode = st.selectbox("OCR Mode", ["Simple OCR", "Structured OCR"])
    doc_type = st.selectbox("Document Type", [x.value for x in DocType])
    llm_choice = st.multiselect(
        "LLM(s) to use",
        ["Mistral", "OpenRouter"],
        default=["Mistral"] if not openrouter_client else ["Mistral","OpenRouter"],
        help="Select one or both.  If OpenRouter key missing, option is disabled.",
    )
    if "OpenRouter" in llm_choice and not openrouter_client:
        st.warning("OpenRouter key missing; deselect or add OPENROUTER_API_KEY.")
        llm_choice = [c for c in llm_choice if c!="OpenRouter"]

    custom_model: BaseModel | None = None
    current_schema: dict | None = None
    if ocr_mode == "Structured OCR":
        if st.checkbox("Use custom JSON schema"):
            txt = st.text_area("Schema (JSON)", height=160)
            if txt.strip():
                try:
                    current_schema = json.loads(txt)
                    custom_model = dynamic_model(current_schema)
                    st.success("✅ Schema OK")
                except (json.JSONDecodeError, ValidationError) as e:
                    st.error(f"Schema error: {e}")

# ---------------- Body ------------------ #
left, right = st.columns(2)
with left:
    st.subheader("📤 Upload")
    uploaded = st.file_uploader("Choose image or PDF", ["jpg","jpeg","png","pdf"])
    run = st.button("🚀 Process", type="primary", disabled=not uploaded)

if run and uploaded:
    with st.spinner("Running OCR → LLMs …"):
        # -- OCR --------------------------------------------------- #
        if Path(uploaded.name).suffix.lower() == ".pdf" or doc_type=="PDF":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            md, preview_url = ocr_pdf(tmp_path, ocr_mode=="Simple OCR")
            os.unlink(tmp_path)
        else:
            md, preview_url = ocr_image(uploaded, ocr_mode=="Simple OCR")

        # -- If only Simple OCR, no LLM required ------------------- #
        results: Dict[str, Any] = {}
        if ocr_mode == "Simple OCR":
            results = {"OCR Markdown": md}
        else:
            model_cls = custom_model or FallbackSchema
            
            # Create the improved prompt
            prompt = _create_extraction_prompt(md, model_cls.schema_json(indent=2))
            
            if "Mistral" in llm_choice:
                chunks = [TextChunk(text=prompt)]
                if preview_url.startswith("data:image"):
                    chunks = [ImageURLChunk(image_url=preview_url)] + chunks
                results["Mistral"] = _mistral_parse(chunks, model_cls)
            if "OpenRouter" in llm_choice:
                img_arg = preview_url if preview_url.startswith("data:image") else None
                results["OpenRouter"] = _openrouter_parse(prompt, img_arg, model_cls)

    # -------------- DISPLAY (side‑by‑side) ----------------------- #
    if ocr_mode == "Simple OCR":
        pdf_col, = st.columns(1)
        with pdf_col:
            st.markdown(md)
    else:
        # Show validation results if we have a schema
        if current_schema and results:
            st.markdown("## 📊 Schema Validation Results")
            
            # Create validation results for each LLM
            validation_results = {}
            for name, json_data in results.items():
                score, details = calculate_schema_compliance_score(json_data, current_schema)
                validation_results[name] = {"score": score, "details": details}
            
            # Display validation summary
            val_cols = st.columns(len(results))
            for idx, (name, result) in enumerate(validation_results.items()):
                with val_cols[idx]:
                    score = result["score"]
                    details = result["details"]
                    
                    # Color-coded score
                    if score >= 90:
                        st.success(f"**{name}**: {score:.1f}% ✅")
                    elif score >= 70:
                        st.warning(f"**{name}**: {score:.1f}% ⚠️")
                    else:
                        st.error(f"**{name}**: {score:.1f}% ❌")
                    
                    # Show details
                    with st.expander(f"{name} Details"):
                        st.write(f"**Schema Valid**: {'✅' if details['overall_valid'] else '❌'}")
                        st.write(f"**Required Fields**: {details['required_fields_present']}/{details['total_required_fields']}")
                        st.write(f"**Optional Fields**: {details['optional_fields_present']}/{details['total_optional_fields']}")
                        
                        if details['validation_errors']:
                            st.write("**Validation Errors**:")
                            for error in details['validation_errors']:
                                st.write(f"• {error}")
            
            # Compare results if multiple LLMs
            if len(results) > 1:
                st.markdown("## 🔍 JSON Comparison")
                llm_names = list(results.keys())
                differences = compare_json_structures(results[llm_names[0]], results[llm_names[1]])
                
                if differences:
                    st.warning(f"Found {len(differences)} differences:")
                    for diff in differences:
                        st.write(f"• {diff}")
                else:
                    st.success("🎉 Both JSONs are identical!")
        
        # Original display with enhanced layout
        st.markdown("## 📋 Results")
        cols = [0.9/(len(results)+1)]*(len(results)+1)   # equal widths
        views = st.columns(cols)

        # 1️⃣ Preview pane
        with views[0]:
            st.markdown("#### 📑 Preview")
            if preview_url.startswith("data:application/pdf"):
                ht = 700
                st.components.v1.html(
                    f'<iframe src="{preview_url}" width="100%" height="{ht}px"></iframe>',
                    height=ht+10
                )
            else:
                st.image(uploaded, use_container_width=True)

        # 2️⃣ One pane per LLM with validation indicators
        for idx, (name, js) in enumerate(results.items(), start=1):
            with views[idx]:
                # Add validation indicator in title
                if current_schema and name in validation_results:
                    score = validation_results[name]["score"]
                    if score >= 90:
                        indicator = "✅"
                    elif score >= 70:
                        indicator = "⚠️"
                    else:
                        indicator = "❌"
                    st.markdown(f"#### {name} JSON {indicator}")
                else:
                    st.markdown(f"#### {name} JSON")
                
                st.json(js)

        # Download options
        st.markdown("## 💾 Download Options")
        download_cols = st.columns(len(results) + 1)
        
        # Download individual results
        for idx, (name, js) in enumerate(results.items()):
            with download_cols[idx]:
                st.download_button(
                    f"📄 {name} JSON", 
                    json.dumps(js, indent=4),
                    file_name=f"ocr_{name.lower()}.json", 
                    mime="application/json"
                )
        
        # Download comparison report
        if current_schema and len(results) > 1:
            with download_cols[-1]:
                # Create comparison report
                report = {
                    "schema_used": current_schema,
                    "results": results,
                    "validation_results": validation_results,
                    "comparison": {
                        "llm_names": list(results.keys()),
                        "differences": compare_json_structures(results[list(results.keys())[0]], results[list(results.keys())[1]]) if len(results) == 2 else []
                    }
                }
                
                st.download_button(
                    "📊 Full Report", 
                    json.dumps(report, indent=4),
                    file_name="ocr_comparison_report.json", 
                    mime="application/json"
                )
