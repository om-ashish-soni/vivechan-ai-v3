from dotenv import dotenv_values
config=dotenv_values()

def get_config(KEY_NAME):
    return config.get(KEY_NAME,None)