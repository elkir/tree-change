
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

def crop(ext,crop):
	xmin,ymin,xmax,ymax =ext
	xmin =xmin+(xmax-xmin)*crop[0]
	ymin =ymin+(ymax-ymin)*crop[2]
	xmax =xmin+(xmax-xmin)*crop[1]
	ymax =ymin+(ymax-ymin)*crop[3]
	return xmin, ymin, xmax, ymax