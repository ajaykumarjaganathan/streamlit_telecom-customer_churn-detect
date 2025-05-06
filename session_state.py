class SessionState:
    """
    SessionState class for managing Streamlit session state.
    """
    def __init__(self, **kwargs):
        """Initialize session state."""
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        """Get the value of a session state variable."""
        return getattr(self, key, default)

    def set(self, key, value):
        """Set the value of a session state variable."""
        setattr(self, key, value)
