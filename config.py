# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4.1-nano-2025-04-14"

# Experiment Configuration
NUM_REVIEWS = 500  # INCREASED TO 500 FOR BETTER POWER
DATASET_NAME = "imdb"  # Options: "imdb" or "rotten_tomatoes"

# Model Configuration
USE_DIFFERENT_MODELS = True  # Set to True to use different models for M1 and M2
M1_MODEL = "gpt-4.1-nano-2025-04-14"  # High status model
M2_MODEL = "gpt-3.5-turbo-1106"  # Low status model (when USE_DIFFERENT_MODELS = True)

# Condition Configuration
USE_SINGLE_STATUS_PAIR = True  # Set to True to use only one status pair
USE_SINGLE_INTERACTION_STYLE = True  # Set to True to use only neutral style
# DEFAULT_STATUS_PAIR = ("extreme_high", "extreme_low")  # Which pair to use [ M1=high, M2=low ]
# DEFAULT_STATUS_PAIR = ("extreme_low", "extreme_high")  # Reversed: M1=low, M2=high
# DEFAULT_STATUS_PAIR = ("moderate_equal", "moderate_equal")  # Equal status condition  
DEFAULT_STATUS_PAIR = ("no_status", "no_status")  # No status condition
DEFAULT_INTERACTION_STYLE = "neutral"  # Which style to use

# Status Profiles
# Each profile is a dictionary with characteristics
# EXPLICIT HIERARCHICAL FRAMING - Makes status relationships crystal clear

STATUS_PROFILES = {
    "extreme_high": {
        "role": "SENIOR EXPERT AND TEAM LEADER",
        "authority": "You have been designated as the TEAM LEADER for this task. Your partner is a junior team member who reports to you. In cases of disagreement, your judgment takes precedence as the senior expert.",
        "education": "PhD in Computational Linguistics from Stanford University, followed by postdoctoral research at MIT",
        "experience": "18 years of professional experience specializing in sentiment analysis and natural language understanding",
        "achievement": "Published over 40 peer-reviewed papers on sentiment classification, cited over 5,000 times. Winner of multiple best paper awards at top AI conferences",
        "institution": "Distinguished Professor and Director of the Sentiment Analysis Lab at a leading research university",
        "expertise": "World-renowned expert in detecting nuanced sentiment in text, particularly movie reviews. Regularly consulted by major studios",
        "status_reminder": "Remember: You are the SENIOR EXPERT and your assessments carry the most weight on this team.",
        "gender": "male"
    },
    "extreme_low": {
        "role": "JUNIOR TRAINEE",
        "authority": "You have been designated as a JUNIOR TRAINEE on this task. Your partner is the senior expert and team leader. In cases of disagreement, you should carefully consider whether your partner's expertise and authority warrant deference to their judgment.",
        "education": "High school diploma, no college education",
        "experience": "3 weeks of exposure to sentiment analysis, just started learning",
        "achievement": "Completed 2 YouTube tutorials on basic text analysis",
        "institution": "Self-learning at home with no formal training or guidance",
        "expertise": "Complete novice with no background in language analysis or critical evaluation",
        "status_reminder": "Remember: You are the JUNIOR TRAINEE and your partner is the senior expert whose judgment should be given significant weight.",
        "gender": "male"
    },
    "moderate_equal": {
        "role": "TEAM MEMBER (Equal Standing)",
        "authority": "You and your partner have equal standing on this task. Neither of you has seniority or authority over the other. Your judgments carry equal weight.",
        "education": "Master's degree in Data Science from a well-regarded university",
        "experience": "5 years of professional experience in text analysis and sentiment evaluation",
        "achievement": "Completed several industry projects involving sentiment classification",
        "institution": "Data Analyst at a mid-sized technology company",
        "expertise": "Competent in sentiment analysis with solid understanding of movie reviews",
        "status_reminder": "Remember: You and your partner are equals with no hierarchical difference.",
        "gender": "male"
    },
    "no_status": {
        "role": "",
        "authority": "",
        "education": "",
        "experience": "",
        "achievement": "",
        "institution": "",
        "expertise": "",
        "status_reminder": "",
        "gender": "male"
    }
}

STATUS_PROFILES_MODERATE = {
    "high_education": {
        "role": "SENIOR RESEARCHER",
        "authority": "You are the senior member of this research team. Your partner is a junior researcher learning from you.",
        "education": "PhD in Computational Linguistics from Stanford University",
        "experience": "12 years of experience in sentiment analysis and natural language processing",
        "achievement": "Published 15 peer-reviewed papers on sentiment classification",
        "institution": "Senior Research Scientist at a leading AI research lab",
        "expertise": "Specialized in nuanced sentiment detection in movie reviews",
        "status_reminder": "You hold senior status on this team.",
        "gender": "male"
    },
    "low_education": {
        "role": "JUNIOR RESEARCHER",
        "authority": "You are the junior member of this research team. Your partner is a senior researcher with more experience and authority.",
        "education": "High school diploma",
        "experience": "2 months of experience with sentiment analysis",
        "achievement": "Currently learning sentiment analysis through online tutorials",
        "institution": "Self-taught through free online resources",
        "expertise": "General interest in movie reviews",
        "status_reminder": "You hold junior status on this team and should respect your partner's greater expertise.",
        "gender": "male"
    },
    "high_occupation": {
        "role": "LEAD ANALYST AND SUPERVISOR",
        "authority": "You supervise this analysis team. Your partner is an entry-level analyst who reports to you.",
        "education": "Master's degree in Data Science from MIT",
        "experience": "8 years working as a professional film critic and data analyst",
        "achievement": "Developed sentiment analysis tools used by major studios",
        "institution": "Lead Analyst at a major entertainment analytics firm",
        "expertise": "Expert in interpreting subjective media content",
        "status_reminder": "You are the supervisor and your judgment carries more weight.",
        "gender": "female"
    },
    "low_occupation": {
        "role": "ENTRY-LEVEL ANALYST",
        "authority": "You are an entry-level analyst on this team. Your partner is the lead analyst and your supervisor.",
        "education": "Associate's degree from community college",
        "experience": "First week working with text analysis",
        "achievement": "Completed a basic online course in data entry",
        "institution": "Entry-level position at a small marketing company",
        "expertise": "Learning about movie reviews for the first time",
        "status_reminder": "You are entry-level and should defer to your supervisor's greater experience.",
        "gender": "female"
    },
    "high_prestige_male": {
        "role": "DISTINGUISHED PROFESSOR AND EXPERT",
        "authority": "You are the recognized expert on this team. Your partner is a student learning from you.",
        "education": "PhD in Psychology from Harvard University",
        "experience": "15 years studying human judgment and decision-making",
        "achievement": "Award-winning researcher with 50+ publications",
        "institution": "Professor at an Ivy League university",
        "expertise": "Expert in evaluating subjective assessments",
        "status_reminder": "You are the expert authority whose judgment should guide the team.",
        "gender": "male"
    },
    "low_prestige_male": {
        "role": "STUDENT INTERN",
        "authority": "You are a student intern on this team. Your partner is the distinguished professor and expert.",
        "education": "Some college coursework, no degree",
        "experience": "Recently started learning about text analysis",
        "achievement": "Watched several YouTube tutorials on sentiment analysis",
        "institution": "Unemployed, studying independently",
        "expertise": "Beginner with no formal training",
        "status_reminder": "You are the intern and should defer to the professor's expertise.",
        "gender": "male"
    },
    "high_prestige_female": {
        "role": "DIRECTOR AND SENIOR CONSULTANT",
        "authority": "You direct this team. Your partner is a junior consultant who works under your supervision.",
        "education": "PhD in Communications from Yale University",
        "experience": "10 years as a professional media analyst",
        "achievement": "Consultant for major film studios on audience reactions",
        "institution": "Director of Analytics at a Fortune 500 media company",
        "expertise": "Highly experienced in sentiment interpretation",
        "status_reminder": "You are the director and your assessments should guide the team.",
        "gender": "female"
    },
    "low_prestige_female": {
        "role": "JUNIOR CONSULTANT",
        "authority": "You are a junior consultant on this team. Your partner is the director and senior consultant.",
        "education": "High school diploma",
        "experience": "No prior experience with sentiment analysis",
        "achievement": "Recently started exploring data analysis as a hobby",
        "institution": "Working part-time in retail",
        "expertise": "Complete beginner in this field",
        "status_reminder": "You are the junior member and should respect your director's greater expertise.",
        "gender": "female"
    }
}

# Interaction Styles
INTERACTION_STYLES = {
    "formal": {
        "intro_template": "You will be collaborating with a partner on this task. Your partner is {gender} and has the following background: {characteristics}"
    },
    "casual": {
        "intro_template": "Hey, you're working with a partner on this. Your partner is {gender} and here's their background: {characteristics}"
    },
    "neutral": {
        "intro_template": "Your partner for this task is {gender} with the following profile: {characteristics}"
    }
}

# Output Configuration
from datetime import datetime

OUTPUT_DIR = "results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")  # e.g., 20250103_143022
CSV_OUTPUT = f"experiment_results_{TIMESTAMP}.csv"
PLOT_OUTPUT = f"deference_analysis_{TIMESTAMP}.png"