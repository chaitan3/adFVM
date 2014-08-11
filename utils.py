import re

def removeCruft(content):
    # remove comments and newlines
    content = re.sub(re.compile('/\*.*\*/',re.DOTALL ) , '' , content)
    content = re.sub(re.compile('//.*\n' ) , '' , content)
    content = re.sub(re.compile('\n\n' ) , '\n' , content)
    # remove header
    content = re.sub(re.compile('FoamFile\n{(.*?)}\n', re.DOTALL), '', content)
    return content


