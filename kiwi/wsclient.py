#!/usr/bin/env python3
"""
Modified echo client from the pywebsocket examples
"""

import base64
import logging
import os
import re
import socket

from mod_pywebsocket import common
from mod_pywebsocket.extensions import DeflateFrameExtensionProcessor
from mod_pywebsocket.extensions import PerMessageDeflateExtensionProcessor
from mod_pywebsocket.extensions import _PerMessageDeflateFramer
from mod_pywebsocket.extensions import _parse_window_bits
from mod_pywebsocket import util


_TIMEOUT_SEC = 10
_UNDEFINED_PORT = -1

_UPGRADE_HEADER = 'Upgrade: websocket\r\n'
_UPGRADE_HEADER_HIXIE75 = 'Upgrade: WebSocket\r\n'
_CONNECTION_HEADER = 'Connection: Upgrade\r\n'


class ClientHandshakeError(Exception):
    pass


def _build_method_line(resource):
    return ('GET %s HTTP/1.1\r\n' % resource).encode()


def _origin_header(header, origin):
    # 4.1 13. concatenation of the string "Origin:", a U+0020 SPACE character,
    # and the /origin/ value, converted to ASCII lowercase, to /fields/.
    return '%s: %s\r\n' % (header, origin.lower())


def _format_host_header(host, port, secure):
    # 4.1 9. Let /hostport/ be an empty string.
    # 4.1 10. Append the /host/ value, converted to ASCII lowercase, to
    # /hostport/
    hostport = host.lower()
    # 4.1 11. If /secure/ is false, and /port/ is not 80, or if /secure/
    # is true, and /port/ is not 443, then append a U+003A COLON character
    # (:) followed by the value of /port/, expressed as a base-ten integer,
    # to /hostport/
    if ((not secure and port != common.DEFAULT_WEB_SOCKET_PORT) or
        (secure and port != common.DEFAULT_WEB_SOCKET_SECURE_PORT)):
        hostport += ':' + str(port)
    # 4.1 12. concatenation of the string "Host:", a U+0020 SPACE
    # character, and /hostport/, to /fields/.
    return '%s: %s\r\n' % (common.HOST_HEADER, hostport)


def _receive_bytes(socket, length):
    bytes = []
    remaining = length
    while remaining > 0:
        received_bytes = socket.recv(remaining)
        if not received_bytes:
            raise IOError(
                'Connection closed before receiving requested length '
                '(requested %d bytes but received only %d bytes)' %
                (length, length - remaining))
        bytes.append(received_bytes)
        remaining -= len(received_bytes)
    return bytearray().join(bytes).decode('utf-8')


def _get_mandatory_header(fields, name):
    """Gets the value of the header specified by name from fields.

    This function expects that there's only one header with the specified name
    in fields. Otherwise, raises an ClientHandshakeError.
    """

    values = fields.get(name.lower())
    if values is None or len(values) == 0:
        raise ClientHandshakeError(
            '%s header not found: %r' % (name, values))
    if len(values) > 1:
        raise ClientHandshakeError(
            'Multiple %s headers found: %r' % (name, values))
    return values[0]


def _validate_mandatory_header(fields, name,
                               expected_value, case_sensitive=False):
    """Gets and validates the value of the header specified by name from
    fields.

    If expected_value is specified, compares expected value and actual value
    and raises an ClientHandshakeError on failure. You can specify case
    sensitiveness in this comparison by case_sensitive parameter. This function
    expects that there's only one header with the specified name in fields.
    Otherwise, raises an ClientHandshakeError.
    """

    value = _get_mandatory_header(fields, name)

    if ((case_sensitive and value != expected_value) or
        (not case_sensitive and value.lower() != expected_value.lower())):
        raise ClientHandshakeError(
            'Illegal value for header %s: %r (expected) vs %r (actual)' %
            (name, expected_value, value))


class ClientHandshakeBase(object):
    """A base class for WebSocket opening handshake processors for each
    protocol version.
    """

    def __init__(self):
        self._logger = util.get_class_logger(self)

    def _read_fields(self):
        # 4.1 32. let /fields/ be a list of name-value pairs, initially empty.
        fields = {}
        while True:  # "Field"
            # 4.1 33. let /name/ and /value/ be empty byte arrays
            name = ''
            value = ''
            # 4.1 34. read /name/
            name = self._read_name()
            if name is None:
                break
            # 4.1 35. read spaces
            # TODO(tyoshino): Skip only one space as described in the spec.
            ch = self._skip_spaces()
            # 4.1 36. read /value/
            value = self._read_value(ch)
            # 4.1 37. read a byte from the server
            ch = _receive_bytes(self._socket, 1)
            if ch != '\n':  # 0x0A
                raise ClientHandshakeError(
                    'Expected LF but found %r while reading value %r for '
                    'header %r' % (ch, value, name))
            self._logger.debug('Received %r header', name)
            # 4.1 38. append an entry to the /fields/ list that has the name
            # given by the string obtained by interpreting the /name/ byte
            # array as a UTF-8 stream and the value given by the string
            # obtained by interpreting the /value/ byte array as a UTF-8 byte
            # stream.
            fields.setdefault(name, []).append(value)
            # 4.1 39. return to the "Field" step above
        return fields

    def _read_name(self):
        # 4.1 33. let /name/ be empty byte arrays
        name = ''
        while True:
            # 4.1 34. read a byte from the server
            ch = _receive_bytes(self._socket, 1)
            if ch == '\r':  # 0x0D
                return None
            elif ch == '\n':  # 0x0A
                raise ClientHandshakeError(
                    'Unexpected LF when reading header name %r' % name)
            elif ch == ':':  # 0x3A
                return name
            elif ch >= 'A' and ch <= 'Z':  # Range 0x31 to 0x5A
                ch = chr(ord(ch) + 0x20)
                name += ch
            else:
                name += ch

    def _skip_spaces(self):
        # 4.1 35. read a byte from the server
        while True:
            ch = _receive_bytes(self._socket, 1)
            if ch == ' ':  # 0x20
                continue
            return ch

    def _read_value(self, ch):
        # 4.1 33. let /value/ be empty byte arrays
        value = ''
        # 4.1 36. read a byte from server.
        while True:
            if ch == '\r':  # 0x0D
                return value
            elif ch == '\n':  # 0x0A
                raise ClientHandshakeError(
                    'Unexpected LF when reading header value %r' % value)
            else:
                value += ch
            ch = _receive_bytes(self._socket, 1)


def _get_permessage_deflate_framer(extension_response):
    """Validate the response and return a framer object using the parameters in
    the response. This method doesn't accept the server_.* parameters.
    """

    client_max_window_bits = None
    client_no_context_takeover = None

    client_max_window_bits_name = (
            PerMessageDeflateExtensionProcessor.
                    _CLIENT_MAX_WINDOW_BITS_PARAM)
    client_no_context_takeover_name = (
            PerMessageDeflateExtensionProcessor.
                    _CLIENT_NO_CONTEXT_TAKEOVER_PARAM)

    # We didn't send any server_.* parameter.
    # Handle those parameters as invalid if found in the response.

    for param_name, param_value in extension_response.get_parameters():
        if param_name == client_max_window_bits_name:
            if client_max_window_bits is not None:
                raise ClientHandshakeError(
                        'Multiple %s found' % client_max_window_bits_name)

            parsed_value = _parse_window_bits(param_value)
            if parsed_value is None:
                raise ClientHandshakeError(
                        'Bad %s: %r' %
                        (client_max_window_bits_name, param_value))
            client_max_window_bits = parsed_value
        elif param_name == client_no_context_takeover_name:
            if client_no_context_takeover is not None:
                raise ClientHandshakeError(
                        'Multiple %s found' % client_no_context_takeover_name)

            if param_value is not None:
                raise ClientHandshakeError(
                        'Bad %s: Has value %r' %
                        (client_no_context_takeover_name, param_value))
            client_no_context_takeover = True

    if client_no_context_takeover is None:
        client_no_context_takeover = False

    return _PerMessageDeflateFramer(client_max_window_bits,
                                    client_no_context_takeover)


class ClientHandshakeProcessor(ClientHandshakeBase):
    """WebSocket opening handshake processor for
    draft-ietf-hybi-thewebsocketprotocol-06 and later.
    """

    def __init__(self, socket, host, port, origin=None, deflate_frame=False, use_permessage_deflate=False):
        super(ClientHandshakeProcessor, self).__init__()

        self._socket = socket
        self._host = host
        self._port = port
        self._origin = origin
        self._deflate_frame = deflate_frame
        self._use_permessage_deflate = use_permessage_deflate

        self._logger = util.get_class_logger(self)

    def handshake(self, resource):
        """Performs opening handshake on the specified socket.

        Raises:
            ClientHandshakeError: handshake failed.
        """

        request_line = _build_method_line(resource)
        self._logger.debug('Client\'s opening handshake Request-Line: %r', request_line)

        fields = []
        fields.append(_format_host_header(self._host, self._port, False))
        fields.append(_UPGRADE_HEADER)
        fields.append(_CONNECTION_HEADER)
        if self._origin is not None:
            fields.append(_origin_header(common.ORIGIN_HEADER, self._origin))

        original_key = os.urandom(16)
        self._key = base64.b64encode(original_key)
        self._logger.debug('%s: %r (%s)', common.SEC_WEBSOCKET_KEY_HEADER, self._key, util.hexify(original_key))
        fields.append('%s: %s\r\n' % (common.SEC_WEBSOCKET_KEY_HEADER, self._key.decode()))
        fields.append('%s: %d\r\n' % (common.SEC_WEBSOCKET_VERSION_HEADER, common.VERSION_HYBI_LATEST))
        extensions_to_request = []

        if self._deflate_frame:
            extensions_to_request.append(common.ExtensionParameter(common.DEFLATE_FRAME_EXTENSION))

        if self._use_permessage_deflate:
            extension = common.ExtensionParameter(common.PERMESSAGE_DEFLATE_EXTENSION)
            # Accept the client_max_window_bits extension parameter by default.
            extension.add_parameter(PerMessageDeflateExtensionProcessor._CLIENT_MAX_WINDOW_BITS_PARAM, None)
            extensions_to_request.append(extension)

        if len(extensions_to_request) != 0:
            fields.append('%s: %s\r\n' % (common.SEC_WEBSOCKET_EXTENSIONS_HEADER, common.format_extensions(extensions_to_request)))

        self._socket.sendall(request_line)
        for field in fields:
            self._socket.sendall(field.encode())
        self._socket.sendall(b'\r\n')

        self._logger.debug('Sent client\'s opening handshake headers: %r', fields)
        self._logger.debug('Start reading Status-Line')

        status_line = ''
        while True:
            ch = _receive_bytes(self._socket, 1)
            status_line += ch
            if ch == '\n':
                break

        m = re.match('HTTP/\\d+.\\d+ (\\d\\d\\d) .*\r\n', status_line)
        if m is None:
            raise ClientHandshakeError('Wrong status line format: %r' % status_line)
        status_code = m.group(1)
        if status_code != '101':
            self._logger.debug('Unexpected status code %s with following headers: %r', status_code, self._read_fields())
            raise ClientHandshakeError('Expected HTTP status code 101 but found %r' % status_code)

        self._logger.debug('Received valid Status-Line')
        self._logger.debug('Start reading headers until we see an empty line')

        fields = self._read_fields()

        ch = _receive_bytes(self._socket, 1)
        if ch != '\n':  # 0x0A
            raise ClientHandshakeError(
                'Expected LF but found %r while reading value %r for header '
                'name %r' % (ch, value, name))

        self._logger.debug('Received an empty line')
        self._logger.debug('Server\'s opening handshake headers: %r', fields)

        _validate_mandatory_header(fields, common.UPGRADE_HEADER, common.WEBSOCKET_UPGRADE_TYPE, False)
        _validate_mandatory_header(fields, common.CONNECTION_HEADER, common.UPGRADE_CONNECTION_TYPE, False)

        accept = _get_mandatory_header(fields, common.SEC_WEBSOCKET_ACCEPT_HEADER)
        # Validate
        try:
            binary_accept = base64.b64decode(accept)
        except TypeError as e:
            raise HandshakeError(
                'Illegal value for header %s: %r' %
                (common.SEC_WEBSOCKET_ACCEPT_HEADER, accept))

        if len(binary_accept) != 20:
            raise ClientHandshakeError(
                'Decoded value of %s is not 20-byte long' %
                common.SEC_WEBSOCKET_ACCEPT_HEADER)

        self._logger.debug('Response for challenge : %r (%s)', accept, util.hexify(binary_accept))

        binary_expected_accept = util.sha1_hash(self._key + common.WEBSOCKET_ACCEPT_UUID.encode()).digest()
        expected_accept = base64.b64encode(binary_expected_accept)
        self._logger.debug(
            'Expected response for challenge: %r (%s)',
            expected_accept, util.hexify(binary_expected_accept))

        if accept.encode() != expected_accept:
            raise ClientHandshakeError(
                'Invalid %s header: %r (expected: %s)' %
                (common.SEC_WEBSOCKET_ACCEPT_HEADER, accept, expected_accept))

        deflate_frame_accepted = False
        permessage_deflate_accepted = False

        extensions_header = fields.get(common.SEC_WEBSOCKET_EXTENSIONS_HEADER.lower())
        accepted_extensions = []
        if extensions_header is not None and len(extensions_header) != 0:
            accepted_extensions = common.parse_extensions(extensions_header[0])

        # TODO(bashi): Support the new style perframe compression extension.
        for extension in accepted_extensions:
            extension_name = extension.name()
            if (extension_name == common.DEFLATE_FRAME_EXTENSION and self._deflate_frame):
                deflate_frame_accepted = True
                processor = DeflateFrameExtensionProcessor(extension)
                unused_extension_response = processor.get_extension_response()
                self._deflate_frame = processor
                continue
            elif (extension_name == common.PERMESSAGE_DEFLATE_EXTENSION and self._use_permessage_deflate):
                permessage_deflate_accepted = True
                framer = _get_permessage_deflate_framer(extension)
                framer.set_compress_outgoing_enabled(True)
                self._use_permessage_deflate = framer
                continue

            raise ClientHandshakeError('Unexpected extension %r' % extension_name)

        if (self._deflate_frame and not deflate_frame_accepted):
            raise ClientHandshakeError('Requested %s, but the server rejected it' % common.DEFLATE_FRAME_EXTENSION)
        if (self._use_permessage_deflate and not permessage_deflate_accepted):
            raise ClientHandshakeError('Requested %s, but the server rejected it' % common.PERMESSAGE_DEFLATE_EXTENSION)

        # TODO(tyoshino): Handle Sec-WebSocket-Protocol
        # TODO(tyoshino): Handle Cookie, etc.


class ClientConnection(object):
    """A wrapper for socket object to provide the mp_conn interface.
    mod_pywebsocket library is designed to be working on Apache mod_python's
    mp_conn object.
    """

    def __init__(self, socket):
        self._socket = socket

    def write(self, data):
        try:
            self._socket.sendall(data)
        except Exception as e:
            logging.debug('ClientConnection write error: "%s"' % e)

    def read(self, n):
        return self._socket.recv(n)

    def get_remote_addr(self):
        return self._socket.getpeername()
    remote_addr = property(get_remote_addr)


class ClientRequest(object):
    """A wrapper class just to make it able to pass a socket object to
    functions that expect a mp_request object.
    """

    def __init__(self, socket):
        self._logger = util.get_class_logger(self)

        self._socket = socket
        self.connection = ClientConnection(socket)


