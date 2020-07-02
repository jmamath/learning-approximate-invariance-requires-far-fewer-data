import vanilla
import invariant_prior
import imgaug
import online_da
import offline_da
from imgaug import augmenters as iaa

if __name__ == '__main__':
	rotation = iaa.Affine(rotate=(-20, 20), name="rotation")
	noise = iaa.AddElementwise((0, 1), name="noise")
	rot_noise = iaa.Sequential([iaa.Affine(rotate=(-20, 20)),
	                           iaa.AddElementwise((0, 1))],name="rotation_noise")

	epochs=400
	epoch_offline=50

	print("VANILLA BNN")
	vanilla.main(epochs)
	print("---------------------------------------")

	print("INVARIANT PRIOR - ROTATION")
	invariant_prior.main(rotation, epochs)
	print("---------------------------------------")

	print("ONLINE DA - ROTATION")
	online_da.main("rotation", epochs)
	print("---------------------------------------")

	print("OFFLINE DA - ROTATION")
	offline_da.main(rotation, epoch_offline)
	print("---------------------------------------")

	print("INVARIANT PRIOR - NOISE")
	invariant_prior.main(noise, epochs)
	print("---------------------------------------")

	print("ONLINE DA - NOISE")
	online_da.main("noise", epochs)
	print("---------------------------------------")

	print("OFFLINE DA - NOISE")
	offline_da.main(noise, epoch_offline)
	print("---------------------------------------")

	print("INVARIANT PRIOR - ROTATION + NOISE")
	invariant_prior.main(rot_noise, epochs)
	print("---------------------------------------")

	print("ONLINE DA - ROTATION + NOISE")
	online_da.main("rot_noise", epochs)
	print("---------------------------------------")

	print("OFFLINE DA - ROTATION + NOISE")
	offline_da.main(rot_noise, epoch_offline)
	print("---------------------------------------")
