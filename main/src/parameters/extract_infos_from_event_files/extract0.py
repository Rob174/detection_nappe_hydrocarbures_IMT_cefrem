from tensorflow.python.summary.summary_iterator import summary_iterator


class Extract0:
    def __init__(self,event_filepath: str):
        event_reader = summary_iterator(event_filepath)
        next(event_reader)
