FROM julia:latest

USER root

#Julia Dependencies
ENV JULIA_PKGDIR /opt/julia
ENV JULIA_DEPOT_PATH /opt/julia

#Make Julia environment read/writable
RUN mkdir -p /opt/julia && chmod 777 /opt/julia
RUN mkdir -p /app && chmod 777 /app

#Install Requirements First (Hastens Container Building)
COPY requirements.jl /app/
RUN julia /app/requirements.jl

#Install the BNN Module
COPY BNN.jl /app/
COPY install_BNN.jl /app/
RUN julia /app/BNN.jl
RUN julia /app/install_BNN.jl

#Copy the rest of the source code
COPY . /app/

#Run the program
CMD bash /app/program.sh
