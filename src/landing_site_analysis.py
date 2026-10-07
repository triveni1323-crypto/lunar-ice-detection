def score_landing_site(ice_probability, slope, shadow, distance_to_crater):
    score = (
        0.5 * ice_probability
        + 0.2 * (1 - slope)
        + 0.2 * shadow
        + 0.1 * (1 - distance_to_crater)
    )
    return score
