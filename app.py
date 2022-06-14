from flask import Flask
import config
from exts import db
from blueprints import fever_bp
from blueprints import user_bp
from flask_migrate import Migrate
from models import User, Article
app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)
app.register_blueprint(fever_bp)
app.register_blueprint(user_bp)
migrate = Migrate(app, db)


if __name__ == '__main__':
    app.run()
