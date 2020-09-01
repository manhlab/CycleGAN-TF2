import tensorflow as tf
from dataset import get_gan_dataset, count_data_items, data_augment
from cycleGAN import CycleGan, Discriminator, Generator
from glob import glob
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='horse2zebra', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
args = parser.parse_args()



def main():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Device:", tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()
    print("Number of replicas:", strategy.num_replicas_in_sync)

    AUTO = tf.data.experimental.AUTOTUNE
    print(tf.__version__)
    BATCH_SIZE = 8
    EPOCHS_NUM = 30
    GCS_PATH = "/home/redbox/code/cycleGAN-TF2/data/monet"
    MONET_FILENAMES = glob(GCS_PATH + "/monet_jpg/*.jpg")
    PHOTO_FILENAMES = glob(GCS_PATH + "/photo_jpg/*.jpg")

    n_monet_samples = count_data_items(MONET_FILENAMES)
    n_photo_samples = count_data_items(PHOTO_FILENAMES)

    print(f"Monet TFRecord files: {len(MONET_FILENAMES)}")
    print(f"Monet image files: {n_monet_samples}")
    print(f"Photo TFRecord files: {len(PHOTO_FILENAMES)}")
    print(f"Photo image files: {n_photo_samples}")
    print(f"Batch_size: {BATCH_SIZE}")
    print(f"Epochs number: {EPOCHS_NUM}")
    
    full_dataset = get_gan_dataset(
        MONET_FILENAMES,
        PHOTO_FILENAMES,
        augment=None,
        repeat=True,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    with strategy.scope():
        monet_generator = Generator()  # transforms photos to Monet-esque paintings
        photo_generator = (
            Generator()
        )  # transforms Monet paintings to be more like photos

        monet_discriminator = (
            Discriminator()
        )  # differentiates real Monet paintings and generated Monet paintings
        photo_discriminator = (
            Discriminator()
        )  # differentiates real photos and generated photos

    with strategy.scope():

        def discriminator_loss(real, generated):
            real_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )(tf.ones_like(real), real)

            generated_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )(tf.zeros_like(generated), generated)

            total_disc_loss = real_loss + generated_loss

            return total_disc_loss * 0.5

    with strategy.scope():

        def generator_loss(generated):
            return tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )(tf.ones_like(generated), generated)

    with strategy.scope():

        def calc_cycle_loss(real_image, cycled_image, LAMBDA):
            loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

            return LAMBDA * loss1

    with strategy.scope():

        def identity_loss(real_image, same_image, LAMBDA):
            loss = tf.reduce_mean(tf.abs(real_image - same_image))
            return LAMBDA * 0.5 * loss

    with strategy.scope():
        monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    with strategy.scope():
        cycle_gan_model = CycleGan(
            monet_generator, photo_generator, monet_discriminator, photo_discriminator
        )

        cycle_gan_model.compile(
            m_gen_optimizer=monet_generator_optimizer,
            p_gen_optimizer=photo_generator_optimizer,
            m_disc_optimizer=monet_discriminator_optimizer,
            p_disc_optimizer=photo_discriminator_optimizer,
            gen_loss_fn=generator_loss,
            disc_loss_fn=discriminator_loss,
            cycle_loss_fn=calc_cycle_loss,
            identity_loss_fn=identity_loss,
        )
    ## Start Training
    cycle_gan_model.fit(
        full_dataset,
        epochs=EPOCHS_NUM,
        steps_per_epoch=(max(n_monet_samples, n_photo_samples) // BATCH_SIZE),
    )


if __name__ == "__main__":

    main()
