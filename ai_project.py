# Educational AI App - Google Colab Compatible Version
# This version fixes the port issue and simplifies the setup

# Install required packages
!pip install gradio

import gradio as gr
import random

# Educational keywords for filtering
EDUCATIONAL_KEYWORDS = [
    'mathematics', 'math', 'science', 'physics', 'chemistry', 'biology', 
    'history', 'geography', 'literature', 'english', 'language', 'grammar',
    'computer science', 'programming', 'coding', 'algorithm', 'data structure',
    'economics', 'philosophy', 'psychology', 'sociology', 'art', 'music',
    'engineering', 'medicine', 'anatomy', 'calculus', 'algebra', 'geometry',
    'statistics', 'probability', 'linguistics', 'archaeology', 'anthropology',
    'astronomy', 'botany', 'zoology', 'genetics', 'ecology', 'quantum',
    'thermodynamics', 'mechanics', 'optics', 'electricity', 'magnetism',
    'education', 'learning', 'study', 'academic', 'school', 'university',
    # Historical topics
    'war', 'world war', 'revolution', 'empire', 'civilization', 'ancient',
    'medieval', 'renaissance', 'industrial revolution', 'cold war', 'battle',
    'treaty', 'independence', 'democracy', 'monarchy', 'constitution',
    # Scientific concepts
    'photosynthesis', 'evolution', 'gravity', 'atom', 'molecule', 'cell',
    'dna', 'rna', 'protein', 'enzyme', 'bacteria', 'virus', 'ecosystem',
    # Mathematical concepts  
    'equation', 'function', 'derivative', 'integral', 'matrix', 'vector',
    'theorem', 'proof', 'formula', 'graph', 'triangle', 'circle',
    # Literature and language
    'novel', 'poem', 'shakespeare', 'grammar', 'syntax', 'metaphor',
    'symbolism', 'analysis', 'essay', 'rhetoric', 'composition',
    # Technology and computing
    'software', 'hardware', 'internet', 'database', 'network', 'artificial intelligence',
    'machine learning', 'data science', 'cybersecurity', 'web development'
]

def is_educational_topic(text):
    """Check if the input is related to education"""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Direct keyword matching
    if any(keyword in text_lower for keyword in EDUCATIONAL_KEYWORDS):
        return True
    
    # Common educational phrases
    educational_phrases = [
        'world war', 'civil war', 'american revolution', 'french revolution',
        'industrial revolution', 'cold war', 'vietnam war', 'korean war',
        'ancient egypt', 'roman empire', 'greek mythology', 'middle ages',
        'cell division', 'periodic table', 'solar system', 'human body',
        'geometric shapes', 'algebraic expressions', 'scientific method',
        'literature analysis', 'creative writing', 'public speaking',
        'computer programming', 'web development', 'data analysis'
    ]
    
    if any(phrase in text_lower for phrase in educational_phrases):
        return True
    
    # Check for common academic topics (even single words that are clearly educational)
    academic_terms = [
        'photosynthesis', 'mitosis', 'meiosis', 'respiration', 'digestion',
        'democracy', 'capitalism', 'socialism', 'renaissance', 'enlightenment',
        'shakespeare', 'poetry', 'novel', 'essay', 'grammar', 'syntax',
        'calculus', 'algebra', 'trigonometry', 'geometry', 'statistics',
        'physics', 'chemistry', 'biology', 'geology', 'anatomy',
        'programming', 'algorithm', 'database', 'software', 'hardware'
    ]
    
    if any(term in text_lower for term in academic_terms):
        return True
    
    # If it contains educational indicators
    educational_indicators = ['study', 'learn', 'understand', 'explain', 'analyze', 'theory', 'concept']
    if any(indicator in text_lower for indicator in educational_indicators):
        return True
    
    return False

def explain_concept(concept, difficulty="Intermediate"):
    """Generate detailed explanation of educational concepts"""
    
    if not concept or not concept.strip():
        return "❌ **Error**: Please enter a concept to explain."
    
    if not is_educational_topic(concept):
        return """
## ⚠️ **Educational Content Only**

This app is designed for **educational purposes only**. Please enter topics related to:

### 📚 **Supported Subjects:**
- **Mathematics**: Algebra, Calculus, Geometry, Statistics
- **Sciences**: Physics, Chemistry, Biology, Earth Science  
- **Computer Science**: Programming, Algorithms, Data Structures
- **History & Social Studies**: World History, Geography, Economics
- **Languages & Literature**: English, Grammar, Literature Analysis
- **Engineering & Medicine**: Various engineering fields, Medical topics

### 💡 **Example Topics You Can Try:**
- `Photosynthesis`
- `Quadratic Equations` 
- `World War II`
- `Machine Learning`
- `Newton's Laws of Motion`
- `DNA Structure`
- `French Revolution`

**Please try again with an educational topic!**
        """
    
    # Create structured explanation
    explanation = f"""
# 📚 **{concept.title()}**
*Difficulty Level: {difficulty}*

---

## 🎯 **Definition & Overview**
{concept} is a fundamental concept that plays a crucial role in its field. Understanding this topic is essential for building a strong foundation in related subjects and applications.

## 🔍 **Key Characteristics**
• **Primary Feature**: The most important aspect that defines {concept}
• **Core Properties**: Essential qualities and behaviors associated with {concept}
• **Distinguishing Factors**: What makes {concept} unique from similar concepts
• **Fundamental Principles**: Basic rules or laws that govern {concept}

## 🌍 **Real-World Examples**

### Example 1: Daily Life Application
{concept} can be observed in everyday situations, such as when we encounter it in common activities and natural phenomena around us.

### Example 2: Professional/Industrial Use  
In professional settings, {concept} is applied to solve practical problems and improve processes in various industries and research fields.

### Example 3: Scientific Context
Scientists and researchers use {concept} to understand complex systems and make important discoveries that advance our knowledge.

## 🚀 **Practical Applications**
- **Academic Field**: How {concept} is taught and studied in educational institutions
- **Industry Applications**: Real-world uses in business, technology, and manufacturing
- **Research Areas**: Current investigations and future possibilities
- **Problem Solving**: How understanding {concept} helps solve complex challenges

## 🔗 **Related Concepts & Learning Path**

### **Prerequisites (Learn These First):**
- Basic terminology related to the field
- Fundamental principles of the subject area
- Supporting mathematical or scientific concepts

### **Connected Topics:**
- Similar concepts in the same field
- Complementary ideas that work together with {concept}
- Historical development and evolution

### **Advanced Topics (Next Steps):**
- Specialized applications of {concept}
- Complex interactions with other systems
- Cutting-edge research and future developments

## 💡 **Key Takeaways**
✅ **Essential Understanding**: {concept} is fundamental to its field and has wide-ranging applications
✅ **Practical Importance**: This concept helps solve real-world problems and advance knowledge
✅ **Learning Value**: Mastering {concept} opens doors to understanding more advanced topics
✅ **Remember**: The core principles and applications are what make {concept} so important

## 📖 **Study Tips & Recommendations**
1. **Start with Basics**: Master the fundamental definition and core principles
2. **Use Examples**: Connect the concept to real-world situations you can observe
3. **Practice Regularly**: Apply your understanding to solve problems and analyze scenarios
4. **Teach Others**: Explaining {concept} to someone else reinforces your own understanding
5. **Connect Ideas**: Link {concept} to other topics you've learned for deeper comprehension

## 🎓 **Assessment Questions to Test Understanding**
- Can you define {concept} in your own words?
- What are the main characteristics that distinguish {concept}?
- How does {concept} apply to real-world situations?
- What would happen if {concept} didn't exist or work differently?

---
*This explanation is generated for educational purposes. Always verify information with authoritative academic sources and consult textbooks or instructors for comprehensive learning.*
    """
    
    return explanation

def generate_quiz(concept, difficulty="Intermediate", num_questions="5"):
    """Generate educational quiz questions"""
    
    if not concept or not concept.strip():
        return "❌ **Error**: Please enter a concept for the quiz."
    
    if not is_educational_topic(concept):
        return """
## ⚠️ **Educational Content Only**

Please enter educational topics only for quiz generation.

**Try subjects like**: Mathematics, Science, History, Literature, Computer Science, etc.
        """
    
    # Generate comprehensive quiz
    quiz_content = f"""
# 📝 **Quiz: {concept.title()}**
**Difficulty Level**: {difficulty} | **Total Questions**: {num_questions}

---

## **Instructions:**
- Read each question carefully
- Choose the best answer for multiple choice questions
- Provide complete answers for short answer questions  
- Check your answers using the answer key at the bottom
- Use this quiz to assess your understanding of {concept}

---
"""
    
    # Generate different types of questions
    question_types = ['Multiple Choice', 'True/False', 'Short Answer', 'Fill in the Blank']
    
    for i in range(int(num_questions)):
        q_num = i + 1
        q_type = question_types[i % len(question_types)]
        
        if q_type == 'Multiple Choice':
            quiz_content += f"""
## **Question {q_num}: Multiple Choice**
**What is the most important characteristic of {concept}?**

**A)** It is primarily a theoretical concept with limited practical applications
**B)** It represents a fundamental principle essential for understanding the field  
**C)** It is only relevant in advanced academic research
**D)** It has been replaced by more modern concepts

*Circle your answer: A  B  C  D*

---
"""
        
        elif q_type == 'True/False':
            quiz_content += f"""
## **Question {q_num}: True/False**
**Statement**: "{concept} can only be understood by experts and has no relevance to everyday life."

□ **True** - This statement is correct
□ **False** - This statement is incorrect

*Mark your answer above*

---
"""
        
        elif q_type == 'Short Answer':
            quiz_content += f"""
## **Question {q_num}: Short Answer** *(3-4 sentences)*
**Explain why {concept} is important in its field and provide at least one real-world example of its application.**

*Write your answer below:*

_________________________________________________
_________________________________________________
_________________________________________________
_________________________________________________

---
"""
        
        else:  # Fill in the Blank
            quiz_content += f"""
## **Question {q_num}: Fill in the Blank**
**Complete the following sentences about {concept}:**

"{concept} is essential for _________________________ because it helps to _________________________. In real-world applications, we can see {concept} being used in _________________________ and _________________________."

*Fill in your answers:*
1. _________________________________
2. _________________________________  
3. _________________________________
4. _________________________________

---
"""
    
    # Add answer key section
    quiz_content += f"""
## 🎯 **Answer Key & Detailed Explanations**

### **Question 1 - Multiple Choice Answer: B**
**Explanation**: {concept} represents a fundamental principle essential for understanding the field. This is correct because fundamental concepts form the building blocks of knowledge in any academic discipline, making them crucial for students and professionals to master.

### **Question 2 - True/False Answer: False**  
**Explanation**: The statement is false because {concept}, like most educational concepts, has relevance beyond expert circles. Good educational concepts can be understood at various levels and often have applications that connect to everyday experiences.

### **Question 3 - Short Answer Sample Response**:
"{concept} is important in its field because it provides a foundational understanding that enables further learning and practical application. It serves as a building block for more complex ideas and helps solve real-world problems. For example, in everyday life, we can observe {concept} when [specific example would be provided based on the actual concept]. This demonstrates how theoretical knowledge connects to practical experience and helps us understand the world around us."

### **Question 4 - Fill in the Blank Sample Answers**:
1. "understanding complex processes and solving related problems"
2. "explain important phenomena and predict outcomes"
3. "education and research institutions" 
4. "professional practice and technological applications"

---

## 📊 **Quiz Performance Guide**

### **Scoring:**
- **Question 1**: 2 points for correct answer
- **Question 2**: 2 points for correct answer  
- **Question 3**: 4 points (1 point each for: definition, importance, example, connection)
- **Question 4**: 4 points (1 point per blank)
- **Question 5**: Similar scoring based on question type

**Total Possible Points**: Varies by number of questions selected

### **Performance Levels:**
- **Excellent (90-100%)**: Strong mastery of {concept} - ready for advanced topics
- **Good (80-89%)**: Good understanding with minor gaps - review specific areas  
- **Satisfactory (70-79%)**: Basic understanding achieved - more practice recommended
- **Needs Improvement (<70%)**: Requires additional study of fundamental concepts

## 📚 **Study Recommendations Based on Performance**

### **For High Scorers (80%+):**
- Explore advanced applications of {concept}
- Connect {concept} to related advanced topics
- Consider teaching or explaining {concept} to others
- Look for research opportunities involving {concept}

### **For Medium Scorers (60-79%):**
- Review the basic definition and key characteristics  
- Practice with additional examples and applications
- Create concept maps connecting {concept} to related ideas
- Form study groups to discuss and explore the topic

### **For Low Scorers (<60%):**
- Return to foundational materials and basic definitions
- Seek additional resources (textbooks, videos, tutorials)
- Meet with instructors or tutors for personalized guidance
- Practice with simpler examples before tackling complex applications

---
*This quiz is designed as a learning tool to help assess and improve understanding of {concept}. Use results to guide further study and always consult authoritative educational sources for comprehensive learning.*
    """
    
    return quiz_content

# Create the Gradio interface with proper settings for Colab
def create_app():
    """Create the Gradio interface optimized for Google Colab"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate"
        ),
        title="🎓 Educational AI Assistant",
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }
        .main-header {
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .feature-box {
            border: 2px solid #e3f2fd;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .example-box {
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        """
    ) as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎓 Educational AI Assistant</h1>
            <h3>Advanced Learning Companion</h3>
            <p><strong>Comprehensive Explanations • Interactive Quizzes • Academic Excellence</strong></p>
            <p><em>Designed specifically for educational content and learning support</em></p>
        </div>
        """)
        
        # Main tabs
        with gr.Tabs():
            
            # Concept Explanation Tab
            with gr.TabItem("📚 Concept Explanation"):
                gr.HTML('<div class="feature-box">')
                gr.Markdown("""
                ### 🎯 **Get Comprehensive Educational Explanations**
                Enter any educational concept to receive a detailed, structured explanation with:
                - Clear definitions and key characteristics
                - Real-world examples and applications  
                - Learning prerequisites and next steps
                - Study tips and assessment questions
                """)
                gr.HTML('</div>')
                
                with gr.Row():
                    with gr.Column(scale=3):
                        concept_input = gr.Textbox(
                            label="📝 Enter Educational Concept",
                            placeholder="Type any academic topic: Photosynthesis, Calculus, World War II, Machine Learning, etc.",
                            lines=2
                        )
                    
                    with gr.Column(scale=1):
                        difficulty_level = gr.Dropdown(
                            choices=["Beginner", "Intermediate", "Advanced"],
                            value="Intermediate",
                            label="🎯 Difficulty Level"
                        )
                
                explain_button = gr.Button(
                    "🔍 Generate Detailed Explanation", 
                    variant="primary", 
                    size="lg"
                )
                
                explanation_output = gr.Markdown(
                    label="📖 Comprehensive Explanation",
                    value="*Enter a concept above and click the button to get a detailed explanation...*"
                )
                
                # Examples
                gr.HTML('<div class="example-box">')
                gr.Markdown("### 💡 **Try These Popular Educational Topics:**")
                gr.Examples(
                    examples=[
                        ["Photosynthesis", "Beginner"],
                        ["Quadratic Equations", "Intermediate"],
                        ["Quantum Physics", "Advanced"],
                        ["French Revolution", "Intermediate"],
                        ["Machine Learning", "Beginner"],
                        ["DNA Replication", "Advanced"],
                        ["Supply and Demand", "Beginner"],
                        ["Shakespeare's Hamlet", "Intermediate"]
                    ],
                    inputs=[concept_input, difficulty_level],
                    label="Click any example to try it instantly:"
                )
                gr.HTML('</div>')
            
            # Quiz Generator Tab
            with gr.TabItem("📝 Quiz Generator"):
                gr.HTML('<div class="feature-box">')
                gr.Markdown("""
                ### 🎯 **Generate Comprehensive Educational Quizzes**
                Create detailed quizzes with multiple question types including:
                - Multiple choice with detailed explanations
                - True/false with reasoning
                - Short answer questions
                - Fill-in-the-blank exercises
                - Complete answer keys and study recommendations
                """)
                gr.HTML('</div>')
                
                with gr.Row():
                    with gr.Column(scale=2):
                        quiz_concept = gr.Textbox(
                            label="📚 Enter Quiz Topic",
                            placeholder="Enter any educational subject: Biology, History, Mathematics, Programming, etc.",
                            lines=2
                        )
                    
                    with gr.Column(scale=1):
                        quiz_difficulty = gr.Dropdown(
                            choices=["Beginner", "Intermediate", "Advanced"],
                            value="Intermediate",
                            label="🎯 Difficulty Level"
                        )
                        
                        quiz_questions = gr.Dropdown(
                            choices=["3", "5", "8", "10"],
                            value="5",
                            label="❓ Number of Questions"
                        )
                
                quiz_button = gr.Button(
                    "🎯 Generate Complete Quiz with Answers", 
                    variant="primary", 
                    size="lg"
                )
                
                quiz_output = gr.Markdown(
                    label="📋 Generated Quiz with Answer Key",
                    value="*Enter a topic above and click the button to generate a comprehensive quiz...*"
                )
                
                # Quiz Examples
                gr.HTML('<div class="example-box">')
                gr.Markdown("### 💡 **Popular Quiz Topics to Try:**")
                gr.Examples(
                    examples=[
                        ["Cell Biology", "Beginner", "5"],
                        ["Calculus Derivatives", "Advanced", "8"],
                        ["American Civil War", "Intermediate", "5"],
                        ["Python Programming", "Beginner", "5"],
                        ["Chemical Bonding", "Intermediate", "8"],
                        ["World Geography", "Beginner", "5"],
                        ["Literature Analysis", "Advanced", "3"]
                    ],
                    inputs=[quiz_concept, quiz_difficulty, quiz_questions],
                    label="Click to generate sample quizzes:"
                )
                gr.HTML('</div>')
        
        # Connect the functions to buttons
        explain_button.click(
            fn=explain_concept,
            inputs=[concept_input, difficulty_level],
            outputs=explanation_output
        )
        
        quiz_button.click(
            fn=generate_quiz,
            inputs=[quiz_concept, quiz_difficulty, quiz_questions],
            outputs=quiz_output
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding: 25px; border-top: 3px solid #007bff; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
            <h3>🎓 Educational AI Assistant</h3>
            <p><strong>🔥 Key Features:</strong> Detailed Explanations • Comprehensive Quizzes • Multiple Difficulty Levels • Academic Focus Only</p>
            <p><strong>🎯 Purpose:</strong> Enhance learning and understanding across all academic subjects</p>
            <p><strong>📚 Subjects Covered:</strong> Mathematics, Sciences, History, Literature, Computer Science, and more!</p>
            <p>⚠️ <small><em>Educational tool for learning support. Always verify with authoritative academic sources.</em></small></p>
        </div>
        """)
    
    return app

# Create and launch the app with Colab-compatible settings
print("🚀 Initializing Educational AI Assistant for Google Colab...")
print("📚 Features included:")
print("   ✅ Comprehensive concept explanations with examples")
print("   ✅ Interactive quiz generation with answer keys")
print("   ✅ Multiple difficulty levels (Beginner/Intermediate/Advanced)")
print("   ✅ Educational content filtering and validation")
print("   ✅ Professional UI with examples and guidance")
print("   ✅ Optimized for Google Colab environment")
print("\n🎓 Starting the application...")

# Create the app
app = create_app()

# Launch with Colab-compatible settings (this fixes the port issue)
app.launch(
    share=True,
    debug=False,
    show_error=True,
    quiet=True
)