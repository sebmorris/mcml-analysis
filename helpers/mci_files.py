create_layer_dict = lambda n, mu_a, mu_s, g, d: {
    "n": n,
    "mu_a": mu_a,
    "mu_s": mu_s,
    "g": g,
    "d": d
}

def createMciString(layers, file_names, number_photons=1e7):
    def createRunString(layers, output_file_name, n_air=1.4, number_photons=1e7, bin_size=0.05, bin_count=100):
        format_layer_string = lambda layer: f"{layer['n']} {layer['mu_a']} {layer['mu_s']} {layer['g']} {layer['d']}"

        separator = "# n, mu_a, mu_s, g, d"
        run = str(output_file_name) + ".mco A\n"
        run += str(int(number_photons)) + '\n' 
        run += f"{bin_size} {bin_size}" + '\n'
        run += f"{bin_count} {bin_count} 1" + '\n'
        run += str(len(layers)) + '\n'
        run += str(n_air) + '\n' + separator + '\n'
        run += '\n'.join([format_layer_string(l) for l in layers]) + '\n'
        run += str(n_air)
        return run

    runStrings = [createRunString(l, f, number_photons=number_photons) for l, f in zip(layers, file_names)]

    output = "1.0 \n" + str(len(layers)) + '\n\n\n'
    output += "\n\n".join(runStrings)

    return output