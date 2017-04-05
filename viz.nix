
{ nixpkgs ? import <nixpkgs> {} }:
let nixpkgs_source = nixpkgs.fetchFromGitHub {
      #owner = "NixOS";
      owner = "jyp";
      repo = "nixpkgs";
      rev = "b94e253e69046e519bcfb04d64bf3864bf912d19";
      sha256 = "1wlja7hi6zn4i5ma3qmbbpn9dvh1q2c1qvvhdriaad8mhni3ikcj";
    }; in
with import nixpkgs_source {};

(pkgs.python35.withPackages (ps: [
  ps.tensorflowWithoutCuda
  ps.Keras
  ps.numpy
  ps.matplotlib
  ps.tkinter
  ps.h5py
  ])).env

