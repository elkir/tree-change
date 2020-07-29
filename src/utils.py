
def bool_to_word(bool):
    """
    Converts Boolean to Yes/No
    """
    if bool == True:
        return "Yes"
    else:
        return "No"


def prepend_tabs(str, n,line_from=0,line_to=None):
    return ("\n" + "\t" * n).join(('\t' * n + str).splitlines()[line_from:line_to])


def format_list(list):
    return ' '.join(map('{:02d}'.format, list))