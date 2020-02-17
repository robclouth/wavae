patch = []

# CREATE WINDOW
patch.append("#N canvas 0 114 414 236 12;")

# CREATE WAVAE OBJECTS
patch.append("#X obj 10 10 wavae_encoder~;")
patch.append("#X obj 10 60 wavae_decoder~;")

# CREATE COMMENT
patch.append("#X text 140 10 WaVAE autoencoder example patch;")
patch.append("#X text 140 30 This patch should be run at 16k@512.;")
patch.append("#X text 140 50 Send it some audio, and mess with;")
patch.append("#X text 140 70 latent dimensions ! Their ranked;")
patch.append("#X text 140 90 from most to least important.;")

# SET BLOCK SIZE
patch.append("#X obj 10 94 loadbang;")
patch.append("#X msg 10 119 set 4096 1 0.5;")
patch.append("#X obj 10 144 block~;")

# CONNECT INPUT OUTPUT
for i in range(16):
    patch.append(f"#X connect 0 {i} 1 {i};")

patch.append("#X connect 7 0 8 0;")
patch.append("#X connect 8 0 9 0;")

print("\n".join(patch))