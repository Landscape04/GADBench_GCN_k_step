# dataset_utils.py
"""
Dataset Utility Functions
"""


def get_available_datasets():
    """Get list of all available datasets"""
    return ['reddit', 'weibo', 'tolokers', 'questions']


def get_dataset_info():
    """Get detailed information about available datasets"""
    return {
        'reddit': {
            'nodes': 10984,
            'description': 'Reddit social network dataset'
        },
        'weibo': {
            'nodes': 8405,
            'description': 'Weibo social network dataset'
        },
        'tolokers': {
            'nodes': 11758,
            'description': 'Tolokers crowdsourcing dataset'
        },
        'questions': {
            'nodes': 48921,
            'description': 'Questions dataset'
        }
    }