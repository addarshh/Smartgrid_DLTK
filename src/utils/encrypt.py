
from BasePath import base_path
from cryptography.fernet import Fernet


def generate_key():
    """
    : Generate a key and save into a file
    """
    key = Fernet.generate_key()
    with open(base_path + "data/keys/smartgrid.key", "wb") as key_file:
        key_file.write(key)
    return


def load_key():
    """
    : Load previously generated key
    """
    return open(base_path + "data/keys/smartgrid.key").read()


def encrypt_message(message):
    """
    : Encrypts message
    """
    key = load_key()
    encoded_message = message.encode()
    f = Fernet(key)
    encrypted_message = f.encrypt(encoded_message)
    return encrypted_message


def decrypt_message(encrypted_message):
    """
    : Decrypts an encrypted message
    """
    key = load_key()
    f = Fernet(key)
    decrypted_message = f.decrypt(encrypted_message)
    message = decrypted_message.decode()
    return message


if __name__ == "__main__":
    encrypted_message = encrypt_message("<enter message to encrypt here>")
    print(encrypted_message)