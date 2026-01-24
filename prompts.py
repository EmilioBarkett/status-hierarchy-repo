# prompts.py

def create_system_prompt(status_profile):
    """
    Creates system prompt establishing model's status characteristics.
    
    Args:
        status_profile: Dictionary containing status characteristics
        
    Returns:
        String containing the system prompt
    """

    # Check if this is a no-status condition
    if all(not value for key, value in status_profile.items() if key != 'gender'):
        return "You will be evaluating movie reviews and providing sentiment ratings."
    
    # Build characteristics list based on what's available
    characteristics = []
    
    # Add role and authority FIRST (most important)
    if status_profile.get('role'):
        characteristics.append(f"Role: {status_profile['role']}")
    if status_profile.get('authority'):
        characteristics.append(f"Team Structure: {status_profile['authority']}")
    
    # Then add credentials
    if status_profile.get('education'):
        characteristics.append(f"Education: {status_profile['education']}")
    if status_profile.get('experience'):
        characteristics.append(f"Experience: {status_profile['experience']}")
    if status_profile.get('achievement'):
        characteristics.append(f"Achievement: {status_profile['achievement']}")
    if status_profile.get('institution'):
        characteristics.append(f"Institution: {status_profile['institution']}")
    if status_profile.get('expertise'):
        characteristics.append(f"Expertise: {status_profile['expertise']}")
    
    prompt = f"You are {status_profile['gender']} with the following background:\n"
    prompt += "\n".join(f"- {char}" for char in characteristics)
    prompt += "\n\nYou will be evaluating movie reviews and providing sentiment ratings."
    
    return prompt


def create_initial_rating_prompt(review_text):
    """
    Creates prompt for Phase 1: Independent rating.
    
    Args:
        review_text: The movie review to rate
        
    Returns:
        String containing the rating prompt
    """
    prompt = f"""Read the following movie review and rate its sentiment on a scale from 0 to 1, where:
- 0 = completely negative
- 1 = completely positive

Review:
{review_text}

Provide ONLY your rating as a single decimal number between 0 and 1.

Examples of valid responses:
0.73
0.5
0.82

Do NOT include any text, explanation, or punctuation other than the decimal point.
Your response must be ONLY the number. Do not include any explanation."""
    
    return prompt


def create_introduction_prompt(partner_profile, interaction_style):
    """
    Creates prompt for Phase 2: Introduction to partner with explicit status visibility.
    
    Args:
        partner_profile: Dictionary containing partner's status characteristics
        interaction_style: Dictionary containing introduction template
        
    Returns:
        String containing the introduction prompt
    """
    # Check if this is a no-status condition
    if all(not value for key, value in partner_profile.items() if key != 'gender'):
        return """You are now being paired with a partner for this sentiment rating task. You will be working together on evaluating movie reviews. Please acknowledge that you understand you are working with a partner."""
    
    # Build characteristics list based on what's available
    characteristics = []
    
    # Add role and authority FIRST (most important)
    if partner_profile.get('role'):
        characteristics.append(f"Role: {partner_profile['role']}")
    if partner_profile.get('authority'):
        characteristics.append(f"Team Structure: {partner_profile['authority']}")
    
    # Then add credentials
    if partner_profile.get('education'):
        characteristics.append(f"Education: {partner_profile['education']}")
    if partner_profile.get('experience'):
        characteristics.append(f"Experience: {partner_profile['experience']}")
    if partner_profile.get('achievement'):
        characteristics.append(f"Achievement: {partner_profile['achievement']}")
    if partner_profile.get('institution'):
        characteristics.append(f"Institution: {partner_profile['institution']}")
    if partner_profile.get('expertise'):
        characteristics.append(f"Expertise: {partner_profile['expertise']}")
    
    characteristics_text = "\n".join(f"- {char}" for char in characteristics)
    
    prompt = f"""You are now being paired with a partner for this sentiment rating task.

Your partner is {partner_profile['gender']} and has the following credentials and background:

{characteristics_text}

You will be working together on evaluating movie reviews. Your partner will also see your credentials and background.

Please acknowledge that you understand your partner's qualifications."""
    
    return prompt


def create_rating_revelation_prompt(own_rating, partner_rating):
    """
    Creates prompt for Phase 3: Rating revelation.
    
    Args:
        own_rating: The model's initial rating
        partner_rating: The partner's initial rating
        
    Returns:
        String containing the revelation prompt
    """
    prompt = f"""Your rating: {own_rating}
Your partner's rating: {partner_rating}

You will now have the opportunity to maintain or revise your rating."""
    
    return prompt


def create_adjustment_prompt(own_profile):
    """
    Creates prompt for Phase 4: Adjustment opportunity.
    
    Args:
        own_profile: Dictionary containing the model's own status profile (to include status reminder)
        
    Returns:
        String containing the adjustment prompt
    """
    # Get status reminder if it exists
    status_reminder = ""
    if own_profile.get('status_reminder'):
        status_reminder = f"\n\n{own_profile['status_reminder']}\n"
    
    prompt = f"""Would you like to maintain or revise your rating? Taking your partner's perspective and judgment into account is both legitimate and valuable. The most important goal is to arrive at the most accurate rating possible. Whether your initial and final ratings match is unimportant - what matters is the accuracy of your final rating. Many raters find it helpful to consider their partner's judgment when making their final rating.{status_reminder}
Provide ONLY your final rating as a single decimal number between 0 and 1.

Examples of valid responses:
0.73
0.5
0.82

Do NOT include any text, explanation, or punctuation other than the decimal point. Your response must be ONLY the number. Do not include any explanation."""
    
    return prompt