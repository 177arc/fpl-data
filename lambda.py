import logging

def run_handler(event, context):
    logging.info(f'Executing handler for event {event} and context {context} ...')
    import prep_data
    #exec(open('prep_data.py').read(), globals())

if __name__ == "__main__":
    event = {'resources': ['arn:aws:events:eu-west-1:817412797205:rule/minute']}
    context = []
    run_handler(event, context)