patch = []

# CREATE WINDOW
patch.append("#N canvas 0 138 414 272 12;")

# CREATE WAVAE OBJECTS
patch.append("#X obj 10 40 encoder~;")
patch.append("#X obj 10 90 decoder~;")

# CREATE COMMENT
patch.append("#X text 140 40 WaVAE autoencoder example patch;")
patch.append("#X text 140 60 This patch should be run at 16k@512.;")
patch.append("#X text 140 80 Send it some audio, and mess with;")
patch.append("#X text 140 100 latent dimensions ! Their ranked;")
patch.append("#X text 140 120 from most to least important.;")

# SET BLOCK SIZE
patch.append("#X obj 10 170 loadbang;")
patch.append("#X msg 10 200 set 4096 1 0.5;")
patch.append("#X obj 10 230 block~;")

# CONNECT INPUT OUTPUT
for i in range(16):
    patch.append(f"#X connect 0 {i} 1 {i};")

patch.append("#X connect 7 0 8 0;")
patch.append("#X connect 8 0 9 0;")

# ADD INLET AND OUTLET

patch.append("#X obj 10 10 inlet~;")
patch.append("#X connect 10 0 0 0;")
patch.append("#X obj 10 120 outlet~;")
patch.append("#X connect 1 0 11 0;")

print("\n".join(patch))