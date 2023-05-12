from ai.ai import AI
from random import random, randint
from app.main import EMBEDDING_MODEL

class TestAI(AI):

    def _create_embedding(self, text):
        """override to fake call to OpenAI
        The structure of the embedding matches tne expected response from OpenAI,
        but the embedding values are randomly generated
        """

        embedding = {
            'object': 'list',
            'data': [
                {
                    'object': 'embedding',
                    'embedding': [
                        (random()*(10**-(randint(2,3))))*((-1)**randint(1,2))
                        for i in range(1536)
                    ],
                    'index': 0
                }
            ],
            'model': EMBEDDING_MODEL,
            'usage': {
                'prompt_tokens': 99999,
                'total_tokens': 99999
            }
        }
        return embedding