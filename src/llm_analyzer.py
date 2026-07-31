"""
LLM Newsletter Analyzer
=======================
Uses Google's Gemini API to curate and format the newsletter content.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def generate_newsletter(posts):
    """
    Sends posts to Gemini for curation and returns HTML newsletter content.
    
    Args:
        posts: List of post dictionaries from reddit_fetcher
        
    Returns:
        Dictionary with 'newsletter' key containing HTML, or None on failure
    """
    # Validate API key early
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        print("   Please add it to your .env file or set it as an environment variable.")
        return None
    
    # Import Google AI library (done here to avoid import errors if not installed)
    try:
        from google.genai import Client, types
    except ImportError:
        print("❌ Error: google-genai package not installed.")
        print("   Run: pip install google-genai")
        return None
    
    # Format posts for the AI
    formatted_content = _format_posts_for_ai(posts)
    
    # System instruction - defines the AI's personality and output format
    system_instruction = """
    You are Robert Armstrong from the Financial Times. You are writing a "Best of the Week" tech digest. 
    If a post has a link to a highly reputable news source (NYT, FT, Guardian, WSJ, TheVerge...) prioritize those.
    If a post has novel ideas on AI or similar, prioritize those.
    
    TONE: Sophisticated, analytical, slightly cynical, and deeply knowledgeable.
    
    CRITICAL FORMATTING INSTRUCTIONS:
    1. Output ONLY raw HTML code for the email body. No markdown (```).
    2. Start directly with the first <h2> tag.
    
    HTML TEMPLATE PER STORY:
    For every story you select, you MUST use this exact HTML structure:
    
    <div style="margin-bottom: 25px;">
        <h3 style="color: #1a1a1a; margin-bottom: 5px;">[Headline]</h3>
        <p style="color: #333; line-height: 1.6;">[Your witty analysis]</p>
        <p style="font-size: 14px; margin-top: 5px;">
            <a href="SOURCE_URL" style="color: #990000; font-weight: bold; text-decoration: none;">[Read Article]</a> 
            <span style="color: #ccc;">|</span>
            <a href="REDDIT_LINK" style="color: #666; text-decoration: none;">[Discuss on Reddit]</a>
        </p>
    </div>

    GUIDELINES:
    - Filter ruthlessly: Pick top 5-7 stories only.
    - If the "SOURCE URL" is the same as the "REDDIT THREAD" (a text-only post), DO NOT include the [Read Article] link. Just show [Discuss on Reddit].
    """

    try:
        print("    🧠 Connecting to Gemini...")
        client = Client(api_key=GEMINI_API_KEY)
        
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite", 
            contents=[formatted_content],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        
        # Clean up any markdown artifacts
        html_content = response.text
        html_content = html_content.replace("```html", "").replace("```", "").strip()
        
        print("    ✅ Newsletter generated successfully!")
        return {'newsletter': html_content}

    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg.upper() or "401" in error_msg:
            print("❌ Invalid API key. Please check your GEMINI_API_KEY.")
        elif "429" in error_msg or "quota" in error_msg.lower():
            print("❌ API quota exceeded. Please check your Gemini billing.")
        else:
            print(f"❌ Error generating newsletter: {e}")
        return None


def _format_posts_for_ai(posts):
    """
    Formats posts into a structured string for the AI to process.
    """
    formatted = ""
    for i, post in enumerate(posts, 1):
        formatted += f"ITEM #{i}: {post['title']}\n"
        formatted += f"SOURCE URL (Article/Link): {post['url']}\n"
        formatted += f"REDDIT THREAD (Comments): {post['reddit_link']}\n"
        if post.get('text'):
            formatted += f"TEXT SNIPPET: {post['text']}\n"
        formatted += "-" * 30 + "\n"
    return formatted
