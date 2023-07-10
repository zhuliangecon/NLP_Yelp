from flask import Flask, render_template, request, jsonify
from RestaurantRecommender import RestaurantRecommender

app = Flask(__name__)
recommender = RestaurantRecommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form.get('keyword')
    zipcode = request.form.get('zipcode')
    radius = int(request.form.get('radius'))
    results = recommender.recommender(keyword, zipcode, radius)
    return jsonify(results)

@app.route('/draw_page')
def new_page():
    return render_template('draw_page.html')

@app.route('/draw', methods=['POST'])
def draw():
    keyword_list = request.form.getlist('keywords[]')
    zipcode = request.form.get('zipcode')
    radius = int(request.form.get('radius'))
    result = recommender.draw(zipcode, radius, keyword_list)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)