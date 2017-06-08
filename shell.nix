
{ nixpkgs ? import <nixpkgs> {} }:
let
   nixpkgs_source = fetchTarball https://github.com/NixOS/nixpkgs/archive/e496841aba2562c64891b73a2cb95c8d0b87a77c.tar.gz; # using .39 version of CUDA
   # nixpkgs_source = fetchTarball "https://d3g5gsiof5omrk.cloudfront.net/nixpkgs/nixpkgs-17.09pre108282.53835c93cb/nixexprs.tar.xz"; # using .66
   # nixpkgs_source = ~/repo/nixpkgs;
   # nixpkgs_source = nixpkgs.fetchFromGitHub {
   #    #owner = "NixOS";
   #    owner = "jyp";
   #    repo = "nixpkgs";
   #    rev = "b94e253e69046e519bcfb04d64bf3864bf912d19";
   #    sha256 = "1wlja7hi6zn4i5ma3qmbbpn9dvh1q2c1qvvhdriaad8mhni3ikcj";
   #  };
in with import nixpkgs_source {};

(pkgs.python35.withPackages (ps: [
  (if stdenv.isDarwin then ps.tensorflowWithoutCuda else ps.tensorflowWithCuda)
  ps.numpy
  ps.Keras
  # ps.pandas
  # ps.statsmodels
  ps.scipy
  ps.h5py
  # ps.notebook # collision with python 2.7
  ])).env

