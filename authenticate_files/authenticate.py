import jwt
import bcrypt
import streamlit as st
from typing import Optional
from datetime import datetime, timedelta
from .hasher import Hasher
from .validator import Validator
from .utils import generate_random_pw
from .exceptions import CredentialsError, DeprecationError, ForgotError, LoginError, RegisterError, ResetError, UpdateError

class Authenticator:
    """
    This class manages user authentication including login, logout, registration, password reset, and forgot password/username.
    """
    def __init__(self, credentials: dict, cookie_name: str, key: str, cookie_expiry_days: float = 30.0, 
                 preauthorized: Optional[list] = None, validator: Optional[Validator] = None):
        """
        Initialize the Authenticator instance.

        Parameters
        ----------
        credentials : dict
            A dictionary containing user credentials.
        cookie_name : str
            The name of the JWT cookie stored on the client's browser for passwordless reauthentication.
        key : str
            The key used to hash the signature of the JWT cookie.
        cookie_expiry_days : float, optional
            The number of days before the reauthentication cookie expires on the client's browser, by default 30.0.
        preauthorized : list, optional
            A list of emails of unregistered users who are authorized to register, by default None.
        validator : Validator, optional
            A Validator object that checks the validity of the username, name, and email fields, by default None.
        """
        self.credentials = credentials
        self.credentials['usernames'] = {key.lower(): value for key, value in credentials['usernames'].items()}
        self.cookie_name = cookie_name
        self.key = key
        self.cookie_expiry_days = cookie_expiry_days
        self.preauthorized = preauthorized
        self.cookie_manager = stx.CookieManager()
        self.validator = validator if validator is not None else Validator()

        # Initialize session state variables
        self._initialize_session_state()

    def _initialize_session_state(self):
        """
        Initialize session state variables.
        """
        st.session_state['name'] = None
        st.session_state['authentication_status'] = None
        st.session_state['username'] = None
        st.session_state['logout'] = None
        st.session_state['failed_login_attempts'] = {}

        # Initialize session state variables for each username
        for username in self.credentials['usernames']:
            if 'logged_in' not in self.credentials['usernames'][username]:
                self.credentials['usernames'][username]['logged_in'] = False

    def _token_encode(self) -> str:
        """
        Encodes the contents of the reauthentication cookie.

        Returns
        -------
        str
            The JWT cookie for passwordless reauthentication.
        """
        return jwt.encode({'username': st.session_state['username'], 'exp_date': self.exp_date}, self.key, algorithm='HS256')

    def _token_decode(self) -> str:
        """
        Decodes the contents of the reauthentication cookie.

        Returns
        -------
        str
            The decoded JWT cookie for passwordless reauthentication.
        """
        try:
            return jwt.decode(self.token, self.key, algorithms=['HS256'])
        except:
            return False

    def _set_exp_date(self) -> str:
        """
        Creates the reauthentication cookie's expiry date.

        Returns
        -------
        str
            The JWT cookie's expiry timestamp in Unix epoch.
        """
        return (datetime.utcnow() + timedelta(days=self.cookie_expiry_days)).timestamp()

    def _check_pw(self) -> bool:
        """
        Checks the validity of the entered password.

        Returns
        -------
        bool
            The validity of the entered password by comparing it to the hashed password on disk.
        """
        return bcrypt.checkpw(self.password.encode(), self.credentials['usernames'][self.username]['password'].encode())

    def _check_cookie(self):
        """
        Checks the validity of the reauthentication cookie.
        """
        self.token = self.cookie_manager.get(self.cookie_name)
        if self.token is not None:
            self.token = self._token_decode()
            if self.token is not False:
                if not st.session_state['logout']:
                    if self.token['exp_date'] > datetime.utcnow().timestamp():
                        if 'username' in self.token:
                            st.session_state['username'] = self.token['username']
                            st.session_state['name'] = self.credentials['usernames'][self.token['username']]['name']
                            st.session_state['authentication_status'] = True
                            self.credentials['usernames'][self.token['username']]['logged_in'] = True

    def _record_failed_login_attempts(self, reset: bool = False):
        """
        Records the number of failed login attempts for a given username.

        Parameters
        ----------
        reset : bool, optional
            If True, reset the failed login attempts for the user to 0, by default False.
        """
        if self.username not in st.session_state['failed_login_attempts']:
            st.session_state['failed_login_attempts'][self.username] = 0
        if reset:
            st.session_state['failed_login_attempts'][self.username] = 0
        else:
            st.session_state['failed_login_attempts'][self.username] += 1

    def _check_credentials(self, inplace: bool = True) -> bool:
        """
        Checks the validity of the entered credentials.

        Parameters
        ----------
        inplace : bool, optional
            If True, store the authentication status in session state, by default True.

        Returns
        -------
        bool
            The validity of entered credentials.
        """
        if isinstance(self.max_concurrent_users, int):
            if self._count_concurrent_users() > self.max_concurrent_users - 1:
                raise(LoginError('Maximum concurrent user limit reached. Please try again later.'))
        if self.username.lower() in self.credentials['usernames']:
            if not self.credentials['usernames'][self.username.lower()]['logged_in']:
                if self._check_pw():
                    if inplace:
                        st.session_state['username'] = self.username
                        st.session_state['name'] = self.credentials['usernames'][self.username.lower()]['name']
                        st.session_state['authentication_status'] = True
                    self.credentials['usernames'][self.username.lower()]['logged_in'] = True
                    return True
                else:
                    self._record_failed_login_attempts()
                    raise(LoginError('Incorrect password.'))
            else:
                self._record_failed_login_attempts()
                raise(LoginError('User already logged in.'))
        else:
            self._record_failed_login_attempts()
            raise(LoginError('Invalid username.'))

    def _count_concurrent_users(self) -> int:
        """
        Counts the number of currently logged-in users.

        Returns
        -------
        int
            The number of currently logged-in users.
        """
        count = 0
        for user in self.credentials['usernames']:
            if self.credentials['usernames'][user]['logged_in']:
                count += 1
        return count

    def login(self, username: str, password: str, passwordless: bool = False, **kwargs) -> bool:
        """
        Logs in a user based on the entered credentials.

        Parameters
        ----------
        username : str
            The username to log in with.
        password : str
            The password to log in with.
        passwordless : bool, optional
            If True, generate a passwordless reauthentication cookie, by default False.

        Returns
        -------
        bool
            The success of the login attempt.
        """
        self.username = username.lower()
        self.password = password
        self.max_concurrent_users = kwargs.get('max_concurrent_users')
        self._record_failed_login_attempts(reset=True)
        self._check_credentials(inplace=True)
        if passwordless:
            self.exp_date = self._set_exp_date()
            self.cookie_manager.set(self.cookie_name, self._token_encode(), self.exp_date)
        return True

    def logout(self):
        """
        Logs out the currently logged-in user.
        """
        st.session_state['logout'] = True
        self.credentials['usernames'][st.session_state['username']]['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['name'] = None
        st.session_state['authentication_status'] = None
        self.cookie_manager.expire(self.cookie_name)

    # Other methods of the Authenticator class...
# Other methods of the Authenticator class...

    def register(self, name: str, username: str, email: str, password: str) -> bool:
        """
        Registers a new user.

        Parameters
        ----------
        name : str
            The name of the user.
        username : str
            The username of the user.
        email : str
            The email address of the user.
        password : str
            The password of the user.

        Returns
        -------
        bool
            The success of the registration attempt.
        """
        if self.validator.validate_name(name) and self.validator.validate_username(username) \
                and self.validator.validate_email(email) and self.validator.validate_password(password):
            if username.lower() not in self.credentials['usernames']:
                hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                self.credentials['usernames'][username.lower()] = {'name': name, 'email': email, 'password': hashed_pw}
                return True
            else:
                raise(RegisterError('Username already exists. Please choose another one.'))
        else:
            raise(RegisterError('Invalid registration details.'))

    def forgot_password(self, email: str) -> str:
        """
        Retrieves the username associated with a given email address.

        Parameters
        ----------
        email : str
            The email address for which to retrieve the username.

        Returns
        -------
        str
            The username associated with the provided email address.
        """
        for username, info in self.credentials['usernames'].items():
            if info['email'] == email:
                return username
        raise(ForgotError('No username associated with the provided email address.'))

    def reset_password(self, username: str, new_password: str) -> bool:
        """
        Resets the password for a given username.

        Parameters
        ----------
        username : str
            The username for which to reset the password.
        new_password : str
            The new password to set for the user.

        Returns
        -------
        bool
            The success of the password reset attempt.
        """
        if username.lower() in self.credentials['usernames']:
            hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
            self.credentials['usernames'][username.lower()]['password'] = hashed_pw
            return True
        else:
            raise(ResetError('Invalid username.'))

    def update_password(self, current_password: str, new_password: str) -> bool:
        """
        Updates the password for the currently logged-in user.

        Parameters
        ----------
        current_password : str
            The current password of the user.
        new_password : str
            The new password to set for the user.

        Returns
        -------
        bool
            The success of the password update attempt.
        """
        if self._check_pw():
            return self.reset_password(st.session_state['username'], new_password)
        else:
            raise(UpdateError('Incorrect current password.'))

    def update_email(self, new_email: str) -> bool:
        """
        Updates the email address for the currently logged-in user.

        Parameters
        ----------
        new_email : str
            The new email address to set for the user.

        Returns
        -------
        bool
            The success of the email update attempt.
        """
        if self.validator.validate_email(new_email):
            self.credentials['usernames'][st.session_state['username']]['email'] = new_email
            return True
        else:
            raise(UpdateError('Invalid email address.'))

    def update_name(self, new_name: str) -> bool:
        """
        Updates the name for the currently logged-in user.

        Parameters
        ----------
        new_name : str
            The new name to set for the user.

        Returns
        -------
        bool
            The success of the name update attempt.
        """
        if self.validator.validate_name(new_name):
            self.credentials['usernames'][st.session_state['username']]['name'] = new_name
            return True
        else:
            raise(UpdateError('Invalid name.'))

    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticates a user with the provided username and password.

        Parameters
        ----------
        username : str
            The username of the user.
        password : str
            The password of the user.

        Returns
        -------
        bool
            The success of the authentication attempt.
        """
        if username.lower() in self.credentials['usernames']:
            hashed_pw = self.credentials['usernames'][username.lower()]['password'].encode()
            if bcrypt.checkpw(password.encode(), hashed_pw):
                return True
        return False

    def is_authenticated(self) -> bool:
        """
        Checks if a user is currently authenticated.

        Returns
        -------
        bool
            True if the user is authenticated, False otherwise.
        """
        return st.session_state.get('username') is not None

    def get_current_username(self) -> str:
        """
        Retrieves the username of the currently logged-in user.

        Returns
        -------
        str
            The username of the currently logged-in user.
        """
        return st.session_state.get('username')

    def get_current_user_info(self) -> dict:
        """
        Retrieves information about the currently logged-in user.

        Returns
        -------
        dict
            A dictionary containing information about the currently logged-in user.
        """
        username = st.session_state.get('username')
        if username:
            return self.credentials['usernames'].get(username)
        else:
            return {}

# End of the Authenticator class
