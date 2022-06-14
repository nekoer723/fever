import wtforms
from wtforms.validators import length


class ClaimForm(wtforms.Form):
    claim = wtforms.StringField(validators=[length(min=1)])
    evidence = wtforms.StringField(validators=[length(min=1)])

