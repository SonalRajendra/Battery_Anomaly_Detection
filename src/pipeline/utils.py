def adjusted_r2(r2, n, k):
    """Calculate the adjusted R-squared."""
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)







