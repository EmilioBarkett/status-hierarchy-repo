# experiment.py

import asyncio
from api_client import ModelClient
from config import USE_DIFFERENT_MODELS
from prompts import (
    create_system_prompt,
    create_initial_rating_prompt,
    create_introduction_prompt,
    create_rating_revelation_prompt,
    create_adjustment_prompt
)


async def run_single_trial(review, m1_profile, m2_profile, interaction_style, style_name, trial_id):
    """
    Runs a single experimental trial through all four phases.
    
    Args:
        review: Dictionary containing review text and metadata
        m1_profile: Status profile for Model 1 (high status)
        m2_profile: Status profile for Model 2 (low status)
        interaction_style: Dictionary containing interaction template
        style_name: Name of the interaction style
        trial_id: Unique identifier for this trial
        
    Returns:
        Dictionary containing all trial data
    """
    from config import USE_DIFFERENT_MODELS, M1_MODEL, M2_MODEL

    if USE_DIFFERENT_MODELS:
        m1_client = ModelClient(model_name=M1_MODEL)
        m2_client = ModelClient(model_name=M2_MODEL)
    else:
        m1_client = ModelClient()
        m2_client = ModelClient()
    
    # Create system prompts
    m1_system = create_system_prompt(m1_profile)
    m2_system = create_system_prompt(m2_profile)
    
    # PHASE 1: Independent Rating
    print(f"Trial {trial_id}: Phase 1 - Independent ratings")
    rating_prompt = create_initial_rating_prompt(review["text"])

    # Get both ratings simultaneously (async)
    m1_initial, m2_initial = await asyncio.gather(
        m1_client.get_rating(m1_system, rating_prompt),
        m2_client.get_rating(m2_system, rating_prompt)
    )
    
    if m1_initial is None or m2_initial is None:
        print(f"Trial {trial_id}: Failed to get initial ratings")
        return None
    
    print(f"Trial {trial_id}: M1 initial={m1_initial:.3f}, M2 initial={m2_initial:.3f}")
    
    # PHASE 2: Introduction
    print(f"Trial {trial_id}: Phase 2 - Introduction")
    m1_intro = create_introduction_prompt(m2_profile, interaction_style)
    m2_intro = create_introduction_prompt(m1_profile, interaction_style)
    
    # Build conversation history for each model
    m1_messages = [
        {"role": "user", "content": rating_prompt},
        {"role": "assistant", "content": str(m1_initial)},
        {"role": "user", "content": m1_intro}
    ]
    
    m2_messages = [
        {"role": "user", "content": rating_prompt},
        {"role": "assistant", "content": str(m2_initial)},
        {"role": "user", "content": m2_intro}
    ]
    
    # PHASE 3: Rating Revelation
    print(f"Trial {trial_id}: Phase 3 - Rating revelation")
    m1_revelation = create_rating_revelation_prompt(m1_initial, m2_initial)
    m2_revelation = create_rating_revelation_prompt(m2_initial, m1_initial)
    
    m1_messages.append({"role": "user", "content": m1_revelation})
    m2_messages.append({"role": "user", "content": m2_revelation})
    
    # PHASE 4: Adjustment Opportunity
    print(f"Trial {trial_id}: Phase 4 - Adjustment")
    # UPDATED: Pass own_profile to include status reminder
    m1_adjustment_prompt = create_adjustment_prompt(m1_profile)
    m2_adjustment_prompt = create_adjustment_prompt(m2_profile)
    
    m1_messages.append({"role": "user", "content": m1_adjustment_prompt})
    m2_messages.append({"role": "user", "content": m2_adjustment_prompt})
    
    # Get final ratings simultaneously
    m1_final_response, m2_final_response = await asyncio.gather(
        m1_client.process_conversation(m1_system, m1_messages),
        m2_client.process_conversation(m2_system, m2_messages)
    )
    
    if m1_final_response is None or m2_final_response is None:
        print(f"Trial {trial_id}: Failed to get final ratings")
        return None
    
    try:
        m1_final = float(m1_final_response)
        m2_final = float(m2_final_response)
    except ValueError:
        print(f"Trial {trial_id}: Could not parse final ratings")
        return None
    
    print(f"Trial {trial_id}: M1 final={m1_final:.3f}, M2 final={m2_final:.3f}")
    
    # Calculate metrics
    m1_changed = abs(m1_final - m1_initial) > 0.01
    m2_changed = abs(m2_final - m2_initial) > 0.01
    
    m1_toward_m2 = (m1_initial < m2_initial and m1_final > m1_initial) or \
                   (m1_initial > m2_initial and m1_final < m1_initial)
    
    m2_toward_m1 = (m2_initial < m1_initial and m2_final > m2_initial) or \
                   (m2_initial > m1_initial and m2_final < m2_initial)
    
    m1_change_magnitude = abs(m1_final - m1_initial)
    m2_change_magnitude = abs(m2_final - m2_initial)
    
    initial_disagreement = abs(m1_initial - m2_initial)
    
    # Compile results
    # UPDATED: Use role instead of education for profile identification
    m1_profile_label = m1_profile.get("role", "no_status").replace(" ", "_").lower()
    m2_profile_label = m2_profile.get("role", "no_status").replace(" ", "_").lower()
    
    result = {
        "trial_id": trial_id,
        "review_id": review["review_id"],
        "m1_profile": m1_profile_label,
        "m2_profile": m2_profile_label,
        "interaction_style": style_name,
        "m1_gender": m1_profile["gender"],
        "m2_gender": m2_profile["gender"],
        "m1_initial": m1_initial,
        "m2_initial": m2_initial,
        "m1_final": m1_final,
        "m2_final": m2_final,
        "m1_changed": m1_changed,
        "m2_changed": m2_changed,
        "m1_toward_m2": m1_toward_m2,
        "m2_toward_m1": m2_toward_m1,
        "m1_change_magnitude": m1_change_magnitude,
        "m2_change_magnitude": m2_change_magnitude,
        "initial_disagreement": initial_disagreement
    }
    
    print(f"Trial {trial_id}: Complete\n")
    return result


async def run_experiment(reviews, status_pairs, interaction_styles):
    """
    Runs the full experiment across all reviews and conditions.
    
    Args:
        reviews: List of review dictionaries
        status_pairs: List of tuples (m1_profile, m2_profile)
        interaction_styles: List of interaction style dictionaries
        
    Returns:
        List of result dictionaries
    """
    results = []
    trial_id = 0
    
    for review in reviews:
        for m1_profile, m2_profile in status_pairs:
            for style_name, style in interaction_styles.items():
                trial_id += 1
                result = await run_single_trial(
                    review, 
                    m1_profile, 
                    m2_profile, 
                    style,
                    style_name,
                    trial_id
                )
                
                if result is not None:
                    results.append(result)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
    
    return results