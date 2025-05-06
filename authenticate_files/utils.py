import string
import bcrypt
import secrets

def generate_random_pw(length: int = 16) -> str:
    """
    Generates a random password and returns the bcrypt hash of the password.

    Parameters
    ----------
    length: int
        The length of the generated password.

    Returns
    -------
    str
        The bcrypt hash of the randomly generated password.
    """
    alphabet = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(alphabet) for i in range(length))
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')