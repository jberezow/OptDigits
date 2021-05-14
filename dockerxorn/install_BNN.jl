current_dir = pwd()
app_dir = "/app"
cd(app_dir)

push!(LOAD_PATH, app_dir)
push!(LOAD_PATH, current_dir)

using BNN
