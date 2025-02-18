"""Utility functions for AUX cloud services."""

from Cryptodome.Cipher import AES


def encrypt_aes_cbc_zero_padding(iv: bytes, key: bytes, data: bytes) -> bytes | None:
    """
    Encrypt data using AES CBC mode with zero padding.

    Args:
        iv: Initialization vector for CBC mode
        key: AES encryption key
        data: Data to encrypt

    Returns:
        Encrypted data as bytes, or None if encryption fails

    """
    try:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        padded_data = data
        padded_data += b"\x00" * (AES.block_size - len(data) % AES.block_size)
        return cipher.encrypt(padded_data)
    except (ValueError, TypeError):
        return None
