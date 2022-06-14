from exts import db


# 定义ORM模型
class Article(db.Model):
    __tablename__ = "article"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    #context = db.Column(db.Text, nullable=False)

    # 外键
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"))

    # orm关系映射 one to many
    author = db.relationship("User", backref="articles")


class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(200), nullable=False)


class UserExtension(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    school = db.Column(db.String(200), nullable=False)
    # 外键
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    # one to one
    # db.backref反向引用时加参数
    user = db.relationship("User", backref=db.backref("extension", uselist=False))
