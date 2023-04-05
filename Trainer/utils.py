
def eval_string(model_type, model_name, wise, ds_name, total, with_wise, no_wise):
    return f"""Model: {model_type}\nModel Name: {model_name}\nWise: {wise}\nData set name: {ds_name}\nTotal:\n{total}With Wise:\n{with_wise}Without Wise:\n{no_wise}"""

def log_string(model_type, model_name, wise, params, notes):
    return f"""Model: {model_type}\nModel Name: {model_name}\nWise: {wise}\n{params}\nNotes: {notes}\n"""