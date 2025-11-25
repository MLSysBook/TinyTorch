"""Tests for the authentication module (tito.core.auth)."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from tito.core.auth import (
    save_credentials,
    load_credentials,
    delete_credentials,
    get_token,
    is_logged_in,
    AuthReceiver,
    CallbackHandler,
    LocalAuthServer,
)


class TestCredentialsStorage:
    """Test credential storage functions."""

    def setup_method(self):
        """Set up temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        # Mock the credentials directory to use our temp dir
        with patch('tito.core.auth.CREDENTIALS_DIR', self.temp_dir):
            self.test_data = {"access_token": "test_token", "refresh_token": "refresh", "user_email": "test@example.com"}

    def test_save_and_load_credentials(self):
        """Test saving and loading credentials."""
        with patch('tito.core.auth.CREDENTIALS_DIR', self.temp_dir):
            save_credentials(self.test_data)
            loaded = load_credentials()
            assert loaded == self.test_data

    def test_load_nonexistent_credentials(self):
        """Test loading when no credentials exist."""
        with patch('tito.core.auth.CREDENTIALS_DIR', self.temp_dir):
            loaded = load_credentials()
            assert loaded is None

    def test_delete_credentials(self):
        """Test deleting credentials."""
        with patch('tito.core.auth.CREDENTIALS_DIR', self.temp_dir):
            save_credentials(self.test_data)
            delete_credentials()
            loaded = load_credentials()
            assert loaded is None

    def test_get_token(self):
        """Test getting access token."""
        with patch('tito.core.auth.CREDENTIALS_DIR', self.temp_dir):
            save_credentials(self.test_data)
            token = get_token()
            assert token == "test_token"

    def test_get_token_no_credentials(self):
        """Test getting token when no credentials exist."""
        with patch('tito.core.auth.CREDENTIALS_DIR', self.temp_dir):
            token = get_token()
            assert token is None

    def test_is_logged_in(self):
        """Test checking login status."""
        with patch('tito.core.auth.CREDENTIALS_DIR', self.temp_dir):
            assert not is_logged_in()
            save_credentials(self.test_data)
            assert is_logged_in()


class TestAuthReceiver:
    """Test the AuthReceiver class."""

    @patch('tito.core.auth.LocalAuthServer')
    @patch('socket.socket')
    @patch('time.sleep')
    @patch('threading.Thread')
    def test_start_server(self, mock_thread_class, mock_sleep, mock_socket_class, mock_server_class):
        """Test starting the auth server."""
        # Mock the server instance
        mock_server = Mock()
        mock_server.server_address = ('127.0.0.1', 54321)
        mock_server_class.return_value = mock_server
        
        # Mock socket for port checking
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 0  # Success
        mock_socket_class.return_value = mock_socket
        
        # Mock thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_thread_class.return_value = mock_thread
        
        receiver = AuthReceiver()
        port = receiver.start()
        
        assert port == 54321
        mock_server_class.assert_called_once()
        receiver.stop()

    @patch('webbrowser.open')
    @patch('time.sleep')
    def test_wait_for_tokens_timeout(self, mock_sleep, mock_open):
        """Test waiting for tokens with timeout."""
        receiver = AuthReceiver()
        # Mock server without auth_data
        receiver.server = Mock()
        receiver.server.auth_data = None

        tokens = receiver.wait_for_tokens(timeout=0.1)
        assert tokens is None

    @patch('webbrowser.open')
    @patch('time.sleep')
    def test_wait_for_tokens_success(self, mock_sleep, mock_open):
        """Test successful token reception."""
        receiver = AuthReceiver()
        test_tokens = {"access_token": "token", "refresh_token": "refresh", "user_email": "user@example.com"}

        # Mock server with auth_data
        receiver.server = Mock()
        receiver.server.auth_data = test_tokens

        with patch('tito.core.auth.save_credentials') as mock_save:
            tokens = receiver.wait_for_tokens(timeout=1)
            assert tokens == test_tokens
            mock_save.assert_called_with(test_tokens)


class TestCallbackHandler:
    """Test the CallbackHandler class."""

    def test_do_get_callback_success(self):
        """Test successful callback handling."""
        # Create handler directly without server initialization
        handler = CallbackHandler.__new__(CallbackHandler)
        
        # Mock the required attributes
        handler.path = "/callback?access_token=test&refresh_token=refresh&email=user@example.com"
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = Mock()
        handler.server = Mock()
        
        with patch('tito.core.auth.save_credentials') as mock_save:
            handler.do_GET()
            assert handler.server.auth_data == {
                "access_token": "test",
                "refresh_token": "refresh",
                "user_email": "user@example.com"
            }
            mock_save.assert_called_once()

    def test_do_get_invalid_path(self):
        """Test handling of invalid callback path."""
        handler = CallbackHandler.__new__(CallbackHandler)
        
        handler.path = "/invalid"
        handler.send_error = Mock()
        handler.server = Mock()
        
        handler.do_GET()
        handler.send_error.assert_called_with(404, "Not Found")


# Integration test example
def test_full_login_flow():
    """Integration test for the full login flow (mocked)."""
    # This would test the entire flow from AuthReceiver to storage
    # In a real scenario, you'd mock the HTTP server and browser
    pass


class TestLoginCommand:
    """Test the LoginCommand behavior."""

    @patch('tito.core.auth.is_logged_in')
    @patch('tito.commands.base.get_console')
    def test_already_logged_in(self, mock_get_console, mock_is_logged_in):
        """Test that login command exits early if already logged in."""
        from tito.commands.login import LoginCommand
        
        mock_is_logged_in.return_value = True
        
        # Mock console
        mock_console = Mock()
        mock_get_console.return_value = mock_console
        
        # Mock config
        mock_config = Mock()
        command = LoginCommand(mock_config)
        
        # Create mock args
        args = Mock()
        args.force = False
        
        result = command.run(args)
        
        assert result == 0
        mock_console.print.assert_called_with("[green]Already logged in to TinyTorch![/green]")


class TestLogoutCommand:
    """Test the LogoutCommand behavior."""

    @patch('tito.commands.login.AuthReceiver')
    @patch('webbrowser.open')
    @patch('time.sleep')
    @patch('tito.commands.login.delete_credentials')
    @patch('tito.commands.base.get_console')
    def test_logout_with_browser(self, mock_get_console, mock_delete, mock_sleep, mock_open, mock_receiver_class):
        """Test that logout command opens browser and deletes credentials."""
        from tito.commands.login import LogoutCommand
        
        # Mock console
        mock_console = Mock()
        mock_get_console.return_value = mock_console
        
        # Mock receiver
        mock_receiver = Mock()
        mock_receiver.start.return_value = 54321
        mock_receiver_class.return_value = mock_receiver
        
        # Mock config
        mock_config = Mock()
        command = LogoutCommand(mock_config)
        
        # Create mock args
        args = Mock()
        
        result = command.run(args)
        
        assert result == 0
        mock_receiver.start.assert_called_once()
        mock_open.assert_called_once_with("http://127.0.0.1:54321/logout")
        mock_delete.assert_called_once()
        mock_console.print.assert_called_with("[green]Successfully logged out of TinyTorch![/green]")