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
    """Generate a default report when OpenAI is not available"""
    
    # Extract just the condition name without confidence
    if "(" in condition:
        condition_name = condition.split("(")[0].strip()
    else:
        condition_name = condition
    
    # Search for the condition in the definitions
    definitions = CONDITION_DEFINITIONS.split("---")
    condition_definition = ""
    
    for definition in definitions:
        if condition_name.lower() in definition.lower():
            condition_definition = definition.strip()
            break
    
    if condition_definition:
        return f"""
# ECG Analysis: {condition_name}

{condition_definition}

## Clinical Impression
The ECG analysis indicates findings consistent with {condition_name}. 
The characteristic features of this condition are present in the analyzed ECG.

## Recommendation
Please correlate these findings with the patient's clinical presentation for a complete assessment.
"""
    else:
        # Default reports for common conditions if not found in definitions
        default_reports = {
            "NORMAL": """
# ECG Analysis: Normal Sinus Rhythm

## Key Findings
- Regular rhythm with normal rate
- Normal P waves preceding each QRS complex
- Normal PR interval, QRS duration, and QT interval
- No ST segment or T wave abnormalities

## Clinical Impression
The ECG shows a normal sinus rhythm without evidence of arrhythmia, conduction abnormalities, or ischemic changes.

## Recommendation
No specific intervention is required based on ECG findings alone.
""",
            
            "Atrial Fibrillation": """
# ECG Analysis: Atrial Fibrillation

## Key Findings
- Irregularly irregular rhythm
- Absence of discrete P waves
- Presence of fibrillatory waves
- Variable R-R intervals

## Clinical Impression
The ECG shows atrial fibrillation characterized by chaotic atrial activity and irregular ventricular response.

## Recommendation
Consider rate control, rhythm control strategies, and anticoagulation based on clinical context and risk factors.
""",
            
            "Atrial Flutter": """
# ECG Analysis: Atrial Flutter

## Key Findings
- Regular or regularly irregular rhythm
- Characteristic "saw-tooth" flutter waves
- Typical atrial rate of approximately 300 bpm
- Variable AV conduction (commonly 2:1)

## Clinical Impression
The ECG shows atrial flutter with organized rapid atrial activity and controlled ventricular response.

## Recommendation
Consider rate control, rhythm control strategies, and anticoagulation similar to atrial fibrillation management.
""",
            
            "Left bundle branch block": """
# ECG Analysis: Left Bundle Branch Block

## Key Findings
- Wide QRS complex (≥120ms)
- Broad, monophasic R waves in leads I, aVL, V5-V6
- Absence of Q waves in leads I, V5-V6
- ST and T wave discordance with QRS direction

## Clinical Impression
The ECG shows a left bundle branch block pattern consistent with conduction delay in the left ventricle.

## Recommendation
Evaluate for underlying structural heart disease, ischemia, or cardiomyopathy that may be associated with LBBB.
""",
            
            "Right bundle branch block": """
# ECG Analysis: Right Bundle Branch Block

## Key Findings
- Wide QRS complex (≥120ms)
- RSR' pattern in right precordial leads (V1-V3)
- Wide S waves in leads I, aVL, V5-V6
- Normal initial ventricular activation

## Clinical Impression
The ECG shows a right bundle branch block pattern consistent with delayed activation of the right ventricle.

## Recommendation
Consider evaluation for underlying conditions such as pulmonary disease, congenital heart disease, or cor pulmonale if clinically indicated.
""",
            
            "1st-degree AV block": """
# ECG Analysis: First-Degree AV Block

## Key Findings
- PR interval prolongation (>200ms)
- Regular rhythm with normal P waves and QRS complexes
- 1:1 AV conduction (each P wave followed by QRS)

## Clinical Impression
The ECG shows first-degree AV block representing delayed conduction through the AV node.

## Recommendation
Usually benign and requires no specific treatment. Consider medication review if PR interval is markedly prolonged.
"""
        }
        
        # Return default report if available, otherwise generic report
        if condition_name in default_reports:
            return default_reports[condition_name]
        else:
            return f"""
# ECG Analysis: {condition_name}

## Key Findings
The ECG analysis suggests findings consistent with {condition_name}.

## Clinical Impression
The ECG characteristics appear to match the diagnostic criteria for {condition_name}.

## Recommendation
Please correlate these findings with the patient's clinical presentation. Consider additional diagnostic testing to confirm this impression.
""" 