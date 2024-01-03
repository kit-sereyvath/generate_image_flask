from flask import Flask, send_file
from keras.models import load_model
from numpy.random import randn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from flask_cors import CORS

app = Flask("Generate")
CORS(app)

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)

    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def save_plot(examples, n):
    for i in range(n * n):

        pyplot.subplot(n, n, 1 + i)

        pyplot.axis('off')

        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')

    pyplot.savefig("filename.png")
    # pyplot.show()

@app.route('/', methods=['GET'])
def welcome():

    model = load_model('generator_model_100.h5')

    latent_points = generate_latent_points(100, 25)

    X = model.predict(latent_points)

    save_plot(X, 5)

    return send_file("filename.png", mimetype='image/png')
    # return "Hello World!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)