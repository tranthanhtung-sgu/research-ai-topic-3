import os
import json
import requests
from PIL import Image
import io
import base64
import openai

# Load condition definitions with better path handling
def get_condition_definitions_path():
    """Get path to condition definitions using multiple fallback options."""
    # Try different possible locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "resources", "condition_definations.txt"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "resources", "condition_definations.txt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "code", "backend", "resources", "condition_definations.txt")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Default to the first path if none exist
    return possible_paths[0]

CONDITION_DEFINITIONS_PATH = get_condition_definitions_path()

def load_condition_definitions():
    """Load the condition definitions from the text file."""
    try:
        with open(CONDITION_DEFINITIONS_PATH, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Warning: Could not load condition definitions: {e}")
        return ""

CONDITION_DEFINITIONS = load_condition_definitions()

def write(condition, figures, report_text, api_key):
    """
    Generate a plain language report using OpenAI
    
    Args:
        condition: Predicted condition
        figures: List of PIL Images or image file paths with ECG visualizations
        report_text: Summary report text
        api_key: OpenAI API key
        
    Returns:
        report: Plain language report as MARKDOWN
    """
    # Prepare the prompt with condition definitions for medical accuracy
    prompt = f"""
    You are an expert cardiologist reviewing an ECG. The AI system has predicted the condition as: {condition}.
    
    Here's the summary report from the ECG analysis:
    {report_text}
    
    Based on these ECG findings and the following medical definitions of ECG conditions:
    
    {CONDITION_DEFINITIONS}
    
    Please provide a detailed, medically accurate report that:
    
    1. Explains why the predicted condition ({condition}) matches the ECG characteristics in the analysis
    2. Identifies the specific ECG features that support this diagnosis
    3. Mentions any potential differential diagnoses that should be considered
    4. Provides clinical context and significance of these findings
    5. Suggests appropriate next steps for confirmation or treatment
    
    Your report should be convincing to another cardiologist by clearly connecting the observed ECG features to the diagnostic criteria. Use medical terminology appropriate for a physician audience.
    
    Format your response as MARKDOWN with appropriate headings and paragraphs.
    """
    
    try:
        # Initialize the OpenAI client with the API key
        client = openai.OpenAI(api_key=api_key)
        
        # Handle the first figure - convert to base64 for OpenAI
        img_str = None
        message_content = [{"type": "text", "text": prompt}]
        
        # Check if the figures are file paths or PIL Images
        if figures and len(figures) > 0:
            first_fig = figures[0]
            
            # If it's a string (file path)
            if isinstance(first_fig, str):
                with open(first_fig, "rb") as img_file:
                    img_data = img_file.read()
                    img_str = base64.b64encode(img_data).decode('utf-8')
            # If it's a PIL Image
            elif hasattr(first_fig, 'save'):
                img_buffer = io.BytesIO()
                first_fig.save(img_buffer, format="PNG")
                img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            else:
                print(f"Warning: Unsupported figure type: {type(first_fig)}")
        
        # If we have a valid image, include it in the API request
        if img_str:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}",
                    "detail": "auto"
                }
            })
        
        # Make the API request using the client
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": message_content}],
            temperature=0.7,
            max_tokens=800  # Increased for more detailed medical explanations
        )
        
        # Extract the response content
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            report = response.choices[0].message.content
            return report
        else:
            print("Error: Invalid or empty response from OpenAI API")
            return generate_default_report(condition)
            
    except Exception as e:
        print(f"Error generating report: {e}")
        return generate_default_report(condition)

def generate_default_report(condition):
    return "Error generating report"