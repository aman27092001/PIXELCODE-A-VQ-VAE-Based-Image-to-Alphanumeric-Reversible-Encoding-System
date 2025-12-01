# # from vqvae_string_codec import demo_train_and_encode
# from vqvae_string_codec_256 import demo_train_and_encode
#
# if __name__ == "__main__":
#     demo_train_and_encode()


# in python REPL or a script
from vqvae_string_codec_256 import train
model = train(num_epochs=30, dataset_root='./images', image_size=256, batch_size=16, num_embeddings=2048, z_dim=128, hidden=256)
