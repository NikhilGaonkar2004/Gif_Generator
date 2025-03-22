import os
import time
import base64
from io import BytesIO

from flask import Flask, request, render_template_string, jsonify
from google import genai
from google.genai import types
from PIL import Image
from loguru import logger

from dotenv import load_dotenv
load_dotenv()

# Configure logger to write to a file in production
logger.add("app_errors.log", rotation="10 MB", level="ERROR")

app = Flask(__name__)

# Load API key from environment variable (ensure it's set in your production environment)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set in environment variables.")
    raise RuntimeError("GEMINI_API_KEY environment variable must be set.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Determine the proper resampling filter based on Pillow version.
try:
    resample_filter = Image.Resampling.LANCZOS
except AttributeError:
    resample_filter = Image.LANCZOS

def generate_frames(prompt, max_retries=3):
    """
    Generate animation frames using the Gemini API.
    Retries if the response contains fewer than 2 frames.
    """
    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries}: Sending request with prompt: {prompt}")
        try:
            response = client.models.generate_content(
                model="models/gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
            )
        except Exception as e:
            logger.error(f"API call failed on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)
            continue

        # Count frames in the response
        frame_count = 0
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    frame_count += 1
        logger.info(f"Received {frame_count} frames in response")
        if frame_count > 1:
            logger.info(f"Successfully received {frame_count} frames on attempt {attempt}")
            return response
        if attempt == max_retries:
            logger.warning(f"Failed to get multiple frames after {max_retries} attempts. Proceeding with {frame_count} frame(s).")
            return response
        logger.warning(f"Only received {frame_count} frame(s). Retrying with enhanced prompt...")
        prompt = f"{prompt} Please create at least 5 distinct frames showing different stages of the animation."
        time.sleep(1)
    return None

def image_to_base64(image):
    """Convert a PIL image to a base64-encoded string."""
    buffered = BytesIO()
    try:
        # Convert image to RGB mode to avoid errors during saving
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        encoded = ""
    finally:
        buffered.close()
    return encoded

@app.errorhandler(Exception)
def handle_exception(e):
    """
    Global error handler to ensure that unexpected exceptions are caught and logged.
    """
    logger.exception("Unhandled exception occurred:")
    return render_template_string('''
        <!DOCTYPE html>
        <html>
          <head>
            <title>Error</title>
            <style>
              body { font-family: Arial, sans-serif; background: #f5f5f5; text-align: center; padding: 50px; }
              .error { color: red; font-size: 20px; }
            </style>
          </head>
          <body>
            <div class="error">An unexpected error occurred. Please try again later.</div>
          </body>
        </html>
    '''), 500

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    frames_base64 = []
    gif_base64 = None

    if request.method == "POST":
        subject = request.form.get("subject", "A cute dancing monkey")[:60]  # Limit to 60 characters
        style = request.form.get("style", "in an 8-bit pixel art style")[:50]   # Limit to 50 characters
        template = "Create an animation by generating multiple frames, showing"
        info = ",keep the size of the images and background consistent in all the frames"
        prompt = f"{template} {subject} {style}"[:300]  # Limit to 300 characters
        
        try:
            response = generate_frames(prompt)
        except Exception as e:
            logger.error(f"Error during frame generation: {e}")
            result = "Error: Unable to generate frames at this time."
            return render_template_string(TEMPLATE, result=result,
                                          frames_base64=frames_base64,
                                          gif_base64=gif_base64)

        frames = []  # To store PIL Image objects

        # Maximum allowed size to reduce memory usage
        max_size = (800, 800)

        # Extract frames from the response
        if response and response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    try:
                        image = Image.open(BytesIO(part.inline_data.data))
                        # Convert to RGBA and resize if necessary to use less memory
                        image = image.convert("RGBA")
                        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                            image.thumbnail(max_size, resample=resample_filter)
                        frames.append(image)
                        frames_base64.append(image_to_base64(image))
                        logger.info("Frame loaded successfully.")
                    except Exception as e:
                        logger.error(f"Error loading frame: {e}")
        else:
            logger.error("No candidates returned in the response")
            result = "Error: No frames generated."

        if frames:
            try:
                # Create animated GIF in memory using optimized frames
                gif_bytes = BytesIO()
                # Convert frames to 'P' mode for GIF optimization
                optimized_frames = [frame.convert('P', palette=Image.ADAPTIVE) for frame in frames]
                optimized_frames[0].save(gif_bytes, format="GIF", save_all=True,
                                          append_images=optimized_frames[1:], duration=500, loop=0, disposal=2)
                gif_bytes.seek(0)
                gif_base64 = base64.b64encode(gif_bytes.getvalue()).decode('utf-8')
                gif_bytes.close()
                result = "success"
                logger.info("Animation successfully created.")
            except Exception as e:
                logger.error(f"Error creating GIF: {e}")
                result = "Error: Unable to create GIF."
        else:
            result = "Error: No frames generated."

    return render_template_string(TEMPLATE, result=result,
                                  frames_base64=frames_base64,
                                  gif_base64=gif_base64)

TEMPLATE = '''
<!DOCTYPE html>
<html>
  <head>
    <title>GIF Generator</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
     /* Original styles remain unchanged */
body {
  font-family: Arial, sans-serif;
  max-width: 900px;
  margin: 0 auto;
  padding: 20px;
  background-color: #f5f5f5;
}
h1 { 
  color: #333; 
  text-align: center; 
  margin-bottom: 20px; 
}
h2 { 
  color: #444; 
  margin-top: 20px; 
}
.grid-container { 
  display: grid; 
  grid-template-columns: 1fr 1fr; 
  gap: 20px; 
}
.form-container, .gif-container, .frames-container {
  background-color: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.form-container { grid-column: 1; }
.gif-container {
  grid-column: 2;
  text-align: center;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}
.frames-container { 
  margin-top: 20px; 
  grid-column: 1 / span 2; 
}
label { 
  display: block; 
  margin-bottom: 5px; 
  font-weight: bold; 
}
input[type="text"] {
  width: 100%;
  padding: 8px;
  margin-bottom: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
}
input[type="submit"] {
  background-color: #4CAF50;
  color: white;
  padding: 10px 15px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  width: 100%;
}
input[type="submit"]:hover { 
  background-color: #45a049; 
}
.frame-gallery { 
  display: flex; 
  flex-wrap: wrap; 
  gap: 10px; 
  justify-content: center; 
}
.frame { 
  border: 1px solid #ddd; 
  padding: 5px; 
  background-color: white; 
}
.frame img { 
  max-width: 120px; 
  height: auto; 
}
.gif-preview img { 
  max-width: 300px; 
  max-height: 300px; 
  height: auto; 
  object-fit: contain; 
}
.download-btn {
  display: inline-block;
  background-color: #008CBA;
  color: white;
  padding: 10px 15px;
  text-decoration: none;
  border-radius: 4px;
  margin-top: 10px;
}
.download-btn:hover { 
  background-color: #007B9A; 
}
.error { 
  color: red; 
  font-weight: bold; 
  text-align: center; 
  padding: 20px; 
}
.loading { 
  display: none; 
  text-align: center; 
  margin-top: 20px; 
}

/* Improved responsive adjustments */
@media (max-width: 768px) {
  .grid-container { 
    grid-template-columns: 1fr; 
  }
  .form-container, .gif-container {
    grid-column: 1;
    margin-bottom: 20px;
  }
  .frames-container {
    grid-column: 1;
  }
  h1 {
    font-size: 24px;
  }
  h2 {
    font-size: 20px;
  }
  body {
    padding: 15px;
  }
}

@media (max-width: 480px) {
  body {
    padding: 10px;
  }
  .form-container, .gif-container, .frames-container {
    padding: 15px;
  }
  h1 {
    font-size: 22px;
    margin-bottom: 15px;
  }
  h2 {
    font-size: 18px;
    margin-top: 15px;
  }
  input[type="submit"] {
    font-size: 14px;
    padding: 8px 12px;
  }
  .download-btn {
    padding: 8px 12px;
    font-size: 14px;
    width: 100%;
    text-align: center;
    box-sizing: border-box;
  }
  .frame img {
    max-width: 100px;
  }
  .gif-preview img {
    max-width: 100%;
  }
  .frame-gallery {
    gap: 5px;
  }
}

@media (max-width: 320px) {
  body {
    padding: 8px;
  }
  .form-container, .gif-container, .frames-container {
    padding: 10px;
  }
  h1 {
    font-size: 20px;
  }
  h2 {
    font-size: 16px;
  }
  input[type="text"], input[type="submit"] {
    padding: 6px 10px;
  }
  .frame img {
    max-width: 80px;
  }
}

    </style>
    <script>
      function showLoading() {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('submitBtn').disabled = true;
        document.getElementById('submitBtn').value = 'Generating...';
        return true;
      }
    </script>
  </head>
  <body>
    <h1>Animated GIF Generator</h1>
    <div class="grid-container">
      <div class="form-container">
        <h2>Create Your Animation</h2>
        <form method="post" onsubmit="return showLoading()">
          <label for="subject">What would you like to animate?</label>
          <input type="text" name="subject" maxlength="60" value="{{ request.form.get('subject', 'A cute dancing monkey') }}" required>
          <label for="style">Style (e.g., pixel art, watercolor, 3D render):</label>
          <input type="text" name="style" maxlength="50" value="{{ request.form.get('style', 'in an 8-bit pixel art style') }}" required>
          <input type="submit" value="Generate GIF" id="submitBtn">
        </form>
        <div id="loading" class="loading">
          <p>Generating your animation... This may take up to 30 seconds.</p>
          <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExa3lndDJxcG5lbmdwbXQ1M3czZDF6ZjlqcmtpenY2aHpnemxmbzMxOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oEjI6SIIHBdRxXI40/giphy.gif" alt="Loading" width="50">
        </div>
      </div>
      {% if result == 'success' and gif_base64 %}
      <div class="gif-container">
        <h2>Your Animation</h2>
        <div class="gif-preview">
          <img src="data:image/gif;base64,{{ gif_base64 }}" alt="Animated GIF">
        </div>
        <a href="data:image/gif;base64,{{ gif_base64 }}" download="animation.gif" class="download-btn">Download GIF</a>
      </div>
      {% elif result and result != 'success' %}
      <div class="gif-container">
        <p class="error">{{ result }}</p>
      </div>
      {% else %}
      <div class="gif-container">
        <h2>Your Animation Will Appear Here</h2>
        <p>Fill out the form and click "Generate GIF" to create your animation.</p>
      </div>
      {% endif %}
      {% if frames_base64 %}
      <div class="frames-container">
        <h2>Individual Frames</h2>
        <div class="frame-gallery">
          {% for frame in frames_base64 %}
            <div class="frame">
              <img src="data:image/png;base64,{{ frame }}" alt="Frame">
            </div>
          {% endfor %}
        </div>
      </div>
      {% endif %}
    </div>
  </body>
</html>
'''

if __name__ == "__main__":
    # Use Gunicorn in production instead of Flask's built-in server
    app.run(host="0.0.0.0", port=8000)
