class Formatter:
    def format(self, value):
        if isinstance(value, list) and isinstance(value[0], str):
            return ",".join(value)
        return value
