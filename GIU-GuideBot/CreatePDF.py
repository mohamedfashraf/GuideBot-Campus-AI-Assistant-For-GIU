from fpdf import FPDF

# Create a PDF object
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

# Add content to the PDF
script_content = """
Script for GuideBot Thesis Defense

1. Title Slide (1 minute)
- "Good morning, everyone. My name is [Your Name], and I am thrilled to present my thesis project, GuideBot: Campus AI Assistant for GIU.
  This project aims to enhance campus navigation, making it easier for students, newcomers, and visitors to find their way around through real-time AI-driven solutions."
- "I would like to thank my supervisors, Prof. Slim Abdennadher, Dr. Nada Sharaf, and Dr. Omar Shehata, for their guidance throughout this project."

2. Table of Contents (30 seconds)
- "Here’s what I’ll cover in my presentation today: the introduction, related work, methodology and implementation, results, and finally, the conclusion and future work."

3. Transition to Introduction (30 seconds)
- "Let’s begin with the introduction, where I’ll provide an overview of GuideBot’s purpose and objectives."

4. Introduction Slide (1 minute)
- "GuideBot is designed to assist users in real-time by providing navigation guidance and improving the campus experience. It leverages artificial intelligence to enhance accessibility for everyone."

5. Objectives Slide (1 minute)
- "Our objectives include: 
  1. Developing a user-friendly campus navigation bot, 
  2. Providing real-time guidance using natural language processing, and 
  3. Building a scalable, efficient, and cost-effective solution."

6. Approach Slide (1.5 minutes)
- "To achieve these objectives, we used a zero-shot classification model (DeBERTa) for intent recognition, developed a lightweight Flask backend for voice interaction, and integrated hardware through Arduino.
  Additionally, we used PyGame to simulate navigation before full hardware deployment."

7. Transition to Related Work (30 seconds)
- "Now, let’s look at existing solutions and how GuideBot addresses the gaps left by them."

8. Related Work (1.5 minutes)
- "Several solutions like JagBot and ZEUS have been developed for campus navigation and voice assistance. However, these solutions often lack scalability or are too costly to implement broadly. GuideBot aims to fill these gaps with a scalable and cost-effective design."

9. Transition to Methodology & Implementation (30 seconds)
- "Next, I’ll walk you through the methodology and implementation, focusing on how GuideBot functions."

10. System Overview of Functionalities (1 minute)
- "This is an overview of GuideBot’s key functionalities. It provides information based on user commands, checks room availability in real-time, and navigates users across the campus."

11. Flow Diagram of User Interaction (1.5 minutes)
- "This flow diagram illustrates the interaction between the user and GuideBot. Commands are issued through voice or text, processed in real-time, and the system responds with navigation guidance or information retrieval."

12. Tools That Have Been Used (1 minute)
- "GuideBot utilizes various technologies: 
  - The frontend is built using HTML, JavaScript, and Tailwind CSS.
  - The backend, developed in Flask, handles commands and integrates gTTS for text-to-speech conversion.
  - NLP is powered by Hugging Face Transformers, and hardware is controlled via PySerial."

13. Frontend with Images (1 minute)
- "The frontend provides a user-friendly interface for both voice and text interactions. These images demonstrate the design’s simplicity and responsiveness."

14. Backend Details (1 minute)
- "The backend processes commands dynamically. DeBERTa is used for intent recognition, while gTTS ensures smooth voice responses. PyDub is used to refine the audio for non-robotic playback."

15. Video of System Workflow in Action (1.5 minutes)
- "Here is a video showcasing the interaction between the user and GuideBot through the touchscreen interface. The user gives a command, and the bot processes it to provide an appropriate response."
- Play Video 1.
- "As you can see, GuideBot efficiently processes commands, delivering accurate and timely responses."

16. Integration with Hardware (1 minute)
- "The hardware integration is achieved using Arduino and ultrasonic sensors. Commands are relayed via PySerial, enabling the bot to detect obstacles and respond accordingly in real-time."

17. Video of Integration: Software and Hardware in Action for Navigation (1.5 minutes)
- "This video demonstrates the integration between software and hardware for navigation. You’ll see how GuideBot processes commands and maneuvers based on real-time inputs."
- Play Video 2.
- "This seamless integration is what makes GuideBot a reliable navigation assistant."

18. Transition to Results (30 seconds)
- "Let’s now review the results of GuideBot’s performance and trials."

19. Results: Command & Navigation Trials (2 minutes)
- "GuideBot demonstrated a navigation accuracy of 95%, with a command response time of 0.7–2 seconds using caching. These metrics highlight the system’s reliability and efficiency in real-world scenarios."

20. Transition to Conclusion & Future Work (30 seconds)
- "Finally, let’s conclude the presentation and discuss the future directions for GuideBot."

21. To Conclude What We Have Done (1 minute)
- "In summary, GuideBot simplifies campus navigation, making it accessible and efficient for all users. It showcases the potential of AI in solving real-world challenges."

22. Future Work (1.5 minutes)
- "Future improvements include:
  1. Supporting multi-floor navigation, 
  2. Adding more diverse command capabilities, 
  3. Deploying multiple robots for larger campuses, and 
  4. Leveraging cloud technologies to improve response time and scalability."

23. Limitations (1 minute)
- "GuideBot does have limitations, including processing delays under heavy loads and a design limited to single-floor navigation. These challenges provide areas to focus on for future iterations."

24. Thank You Slide (30 seconds)
- "Thank you for your attention. I am happy to answer any questions you may have."

25. References (Optional)
- If required, briefly mention key references or simply transition to Q&A.
"""

# Replace problematic Unicode characters
script_content = (
    script_content.replace("’", "'")
    .replace("“", '"')
    .replace("”", '"')
    .replace("–", "-")
)

# Add the content line by line
for line in script_content.split("\n"):
    pdf.multi_cell(0, 10, line)

# Save the PDF
output_path = "GIU-GuideBot.pdf"
pdf.output(output_path)

print(f"PDF saved to: {output_path}")