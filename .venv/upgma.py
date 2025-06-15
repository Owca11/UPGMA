import numpy as np
from collections import defaultdict
from Bio import Phylo
import io  # Required for saving tree to string/file
import matplotlib.pyplot as plt  # Ensure this is imported


# --- Node Class for UPGMA Tree ---
class Node:
    def __init__(self, name, left=None, right=None, branch_length_to_parent=0.0):
        self.name = name
        self.left = left
        self.right = right
        self.branch_length_to_parent = branch_length_to_parent
        self.height = 0.0  # Height from this node to its furthest leaf

    def is_leaf(self):
        return self.left is None and self.right is None

    def __str__(self):
        return f"{self.name} (len:{self.branch_length_to_parent:.2f})"

    def get_leaves(self):
        if self.is_leaf():
            return [self.name]
        leaves = []
        if self.left:
            leaves.extend(self.left.get_leaves())
        if self.right:
            leaves.extend(self.right.get_leaves())
        return leaves


# --- Provided Tree Drawing Functions ---
def print_tree(node, level=0, last=True, prefix="", file=None):
    if not node:
        return

    connector = "|-- " if not last else "\\-- "
    print(prefix + connector + str(node), file=file)
    new_prefix = prefix + ("    " if last else "|   ")

    if node.right:
        print_tree(node.right, level + 1, False, new_prefix, file=file)
    if node.left:
        print_tree(node.left, level + 1, True, new_prefix, file=file)


def convert_to_biophylo_tree(custom_tree_node):
    if custom_tree_node.is_leaf():
        clade = Phylo.BaseTree.Clade(name=custom_tree_node.name)
        clade.branch_length = custom_tree_node.branch_length_to_parent
    else:
        clade = Phylo.BaseTree.Clade(name=custom_tree_node.name)
        clade.branch_length = custom_tree_node.branch_length_to_parent

        if custom_tree_node.left:
            left_clade = convert_to_biophylo_tree(custom_tree_node.left)
            clade.clades.append(left_clade)
        if custom_tree_node.right:
            right_clade = convert_to_biophylo_tree(custom_tree_node.right)
            clade.clades.append(right_clade)

    return clade


# --- Sequence Loading Functions ---
def load_sequences_from_file(filepath):
    """
    Loads sequences from a FASTA-like file.
    """
    sequences = {}
    current_name = None
    current_sequence = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_name:
                        sequences[current_name] = "".join(current_sequence)
                    current_name = line[1:].strip()
                    current_sequence = []
                else:
                    current_sequence.append(line)
            if current_name:  # Add the last sequence
                sequences[current_name] = "".join(current_sequence)
        print(f"Loaded {len(sequences)} sequences from {filepath}")
        return sequences
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading sequences: {e}")
        return None


def get_sequences_from_user_input():
    """
    Allows user to input sequences directly.
    """
    sequences = {}
    print("Enter sequences. Type 'done' on a new line when finished.")
    while True:
        name = input("Enter sequence name (or 'done'): ").strip()
        if name.lower() == 'done':
            break
        seq = input(f"Enter sequence for {name}: ").strip()
        if seq:
            sequences[name] = seq
        else:
            print("Sequence cannot be empty. Please try again.")
    return sequences


# --- Distance Matrix Calculation (Placeholder) ---
def calculate_distance_matrix(sequences, method="pairwise_alignment"):
    """
    Calculates a distance matrix from a dictionary of sequences.
    YOU MUST REPLACE THIS WITH YOUR ACTUAL ALIGNMENT/DISTANCE CALCULATION.
    """
    if not sequences:
        print("No sequences provided to calculate distance matrix.")
        return None, None

    names = list(sequences.keys())
    num_seq = len(names)
    distance_matrix = np.zeros((num_seq, num_seq))

    print(f"Calculating distance matrix using dummy '{method}' method...")

    for i in range(num_seq):
        for j in range(i + 1, num_seq):
            seq1 = sequences[names[i]]
            seq2 = sequences[names[j]]

            max_len = max(len(seq1), len(seq2))
            s1_padded = seq1.ljust(max_len, '-')
            s2_padded = seq2.ljust(max_len, '-')

            diff = sum(1 for a, b in zip(s1_padded, s2_padded) if a != b)
            dist = diff / max_len if max_len > 0 else 0
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetrical matrix

    print("Distance Matrix Calculated:")
    print("  " + " ".join(f"{name:<5}" for name in names))
    for i, row in enumerate(distance_matrix):
        print(f"{names[i]:<2} " + " ".join(f"{d:<5.2f}" for d in row))

    return distance_matrix, names


def get_distance_matrix_from_user_input():
    """
    Allows user to input a distance matrix directly.
    """
    print("Enter the distance matrix. Enter row by row, space-separated values.")
    print("Example for 3 sequences (A, B, C):")
    print("0.0 0.5 0.8")
    print("0.5 0.0 0.6")
    print("0.8 0.6 0.0")

    names_input = input("Enter sequence names, space-separated (e.g., A B C): ").strip()
    if not names_input:
        print("Names cannot be empty.")
        return None, None
    names = names_input.split()
    num_seq = len(names)

    matrix_rows = []
    for i in range(num_seq):
        while True:
            row_str = input(f"Enter row {i + 1} for {names[i]} (space-separated {num_seq} values): ").strip()
            try:
                row_values = [float(x) for x in row_str.split()]
                if len(row_values) != num_seq:
                    print(f"Error: Expected {num_seq} values, got {len(row_values)}. Please re-enter.")
                    continue
                matrix_rows.append(row_values)
                break
            except ValueError:
                print("Error: Invalid input. Please enter numeric values separated by spaces.")

    distance_matrix = np.array(matrix_rows)

    # Basic correctness check: ensure it's square and reasonably symmetric
    if not np.allclose(distance_matrix, distance_matrix.T):
        print("Warning: The entered matrix is not perfectly symmetric. UPGMA assumes symmetry.")
    if not np.all(np.diag(distance_matrix) == 0):
        print("Warning: Diagonal elements (self-distance) should be zero.")

    print("\nReceived Distance Matrix:")
    print("  " + " ".join(f"{name:<5}" for name in names))
    for i, row in enumerate(distance_matrix):
        print(f"{names[i]:<2} " + " ".join(f"{d:<5.2f}" for d in row))
    return distance_matrix, names


# --- UPGMA Algorithm Implementation ---
def upgma(distance_matrix, names):
    """
    Implements the UPGMA (Unweighted Pair Group Method with Arithmetic Mean) algorithm.
    """
    if not names or distance_matrix is None or len(names) != distance_matrix.shape[0]:
        print("Invalid input for UPGMA algorithm.")
        return None

    clusters = {name: Node(name) for name in names}
    cluster_sizes = {name: 1 for name in names}

    current_distances = distance_matrix.copy()
    current_names = list(names)

    while len(current_names) > 1:
        min_dist = float('inf')
        cluster_i, cluster_j = -1, -1

        for i in range(len(current_names)):
            for j in range(i + 1, len(current_names)):
                if current_distances[i, j] < min_dist:
                    min_dist = current_distances[i, j]
                    cluster_i, cluster_j = i, j

        name1 = current_names[cluster_i]
        name2 = current_names[cluster_j]

        node1 = clusters[name1]
        node2 = clusters[name2]

        size1 = cluster_sizes[name1]
        size2 = cluster_sizes[name2]

        new_node_height = min_dist / 2.0

        node1.branch_length_to_parent = new_node_height - node1.height
        node2.branch_length_to_parent = new_node_height - node2.height

        new_cluster_name = f"({name1},{name2})"
        new_node = Node(new_cluster_name, left=node1, right=node2)
        new_node.height = new_node_height

        clusters[new_cluster_name] = new_node
        cluster_sizes[new_cluster_name] = size1 + size2

        print(f"\nMerging {name1} (size {size1}) and {name2} (size {size2}) at height {new_node_height:.2f}")

        new_num_clusters = len(current_names) - 1
        new_dist_matrix = np.zeros((new_num_clusters, new_num_clusters))
        new_cluster_names_list = []

        k = 0
        for i in range(len(current_names)):
            if i == cluster_i or i == cluster_j:
                continue

            old_cluster_name_k = current_names[i]
            new_cluster_names_list.append(old_cluster_name_k)

            dist_to_new_cluster = (current_distances[i, cluster_i] * size1 +
                                   current_distances[i, cluster_j] * size2) / (size1 + size2)

            new_dist_matrix[k, new_num_clusters - 1] = dist_to_new_cluster
            new_dist_matrix[new_num_clusters - 1, k] = dist_to_new_cluster
            k += 1

        old_to_new_map = {name: idx for idx, name in enumerate(new_cluster_names_list)}
        for i in range(len(current_names)):
            if i == cluster_i or i == cluster_j:
                continue
            for j in range(len(current_names)):
                if j == cluster_i or j == cluster_j:
                    continue
                if i < j:
                    try:
                        new_i = old_to_new_map[current_names[i]]
                        new_j = old_to_new_map[current_names[j]]
                        if new_i < new_j:
                            new_dist_matrix[new_i, new_j] = current_distances[i, j]
                            new_dist_matrix[new_j, new_i] = current_distances[i, j]
                        else:
                            new_dist_matrix[new_j, new_i] = current_distances[i, j]
                            new_dist_matrix[new_i, new_j] = current_distances[i, j]
                    except KeyError:
                        pass

        new_cluster_names_list.append(new_cluster_name)

        current_distances = new_dist_matrix
        current_names = new_cluster_names_list

        print("Updated Clusters:", current_names)
        print("Updated Distance Matrix:")
        print("  " + " ".join(f"{name:<5}" for name in current_names))
        for i, row in enumerate(current_distances):
            print(f"{current_names[i]:<2} " + " ".join(f"{d:<5.2f}" for d in row))

    root_name = current_names[0]
    root_node = clusters[root_name]
    root_node.branch_length_to_parent = 0.0

    return root_node


# --- Main Program Logic ---
def main():

    sequences = {}
    distance_matrix = None
    names = []

    # Input Data Option
    while True:
        print("\nChoose input data option:")
        print("a) Provide a set of sequences")
        print("b) Provide a distance matrix directly")
        choice = input("Enter 'a' or 'b': ").strip().lower()

        if choice == 'a':
            print("\nHow would you like to load sequences?")
            print("1) From a FASTA file")
            print("2) Enter manually")
            seq_choice = input("Enter '1' or '2': ").strip()
            if seq_choice == '1':
                filepath = input("Enter the path to the FASTA file: ").strip()
                sequences = load_sequences_from_file(filepath)
            elif seq_choice == '2':
                sequences = get_sequences_from_user_input()
            else:
                print("Invalid choice. Please enter '1' or '2'.")
                continue

            if sequences:
                names = list(sequences.keys())
                if len(names) < 2:
                    print("Error: UPGMA requires at least two sequences. Please provide more.")
                    sequences = {}
                    continue
                # Calculate distance matrix using MSA or global pairwise alignment results
                distance_matrix, names = calculate_distance_matrix(sequences, method="your_MSA_or_pairwise_alignment")
                if distance_matrix is None:
                    print("Failed to calculate distance matrix. Please check sequence input.")
                    continue
                break
            else:
                print("No sequences loaded. Please try again.")
                continue

        elif choice == 'b':
            distance_matrix, names = get_distance_matrix_from_user_input()
            if distance_matrix is not None and len(names) > 1:
                if distance_matrix.shape[0] != distance_matrix.shape[1] or distance_matrix.shape[0] != len(names):
                    print("Error: Distance matrix dimensions do not match the number of names provided.")
                    distance_matrix = None
                    names = []
                    continue
                break
            else:
                print("Invalid distance matrix or names provided. Please try again.")
                continue
        else:
            print("Invalid choice. Please enter 'a' or 'b'.")

    # --- Perform UPGMA calculation ---
    if distance_matrix is not None and names:
        print("\n--- Starting UPGMA Tree Construction ---")
        upgma_tree_root = upgma(distance_matrix, names)

        if upgma_tree_root:
            print("\n--- UPGMA Tree Constructed Successfully! ---")
            print("\nTextual Representation of the Tree:")
            print_tree(upgma_tree_root)

            # Save the tree and input data to a file
            output_filename = "upgma_tree_output.txt"
            with open(output_filename, 'w') as f:
                f.write("--- UPGMA Phylogenetic Tree Analysis ---\n\n")
                f.write("Input Data:\n")
                if sequences:
                    f.write("Sequences:\n")
                    for name, seq in sequences.items():
                        f.write(f">{name}\n{seq}\n")
                else:
                    f.write("Distance Matrix:\n")
                    f.write("  " + " ".join(f"{name:<5}" for name in names) + "\n")
                    for i, row in enumerate(distance_matrix):
                        f.write(f"{names[i]:<2} " + " ".join(f"{d:<5.2f}" for d in row) + "\n")

                f.write("\n--- Textual Tree Output ---\n")
                tree_output_buffer = io.StringIO()
                print_tree(upgma_tree_root, file=tree_output_buffer)
                f.write(tree_output_buffer.getvalue())
                f.write("\n\n--- End of Tree Output ---\n")
            print(f"\nTree and input data saved to '{output_filename}'")

            # Graphical Representation of the Tree
            print("\n--- Generating Graphical Tree Representation ---")
            try:
                biophylo_root = convert_to_biophylo_tree(upgma_tree_root)
                biophylo_tree = Phylo.BaseTree.Tree(root=biophylo_root)

                # Save to Newick format
                newick_filename = "upgma_tree.nwk"
                Phylo.write(biophylo_tree, newick_filename, "newick")
                print(f"Graphical tree data saved in Newick format to '{newick_filename}'.")
                print("You can visualize this file using tools like Archaeopteryx, FigTree, or online viewers.")

                # Optional: Matplotlib visualization (requires matplotlib and Biopython's matplotlib support)
                try:
                    # Clear any existing figures to ensure a fresh plot
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(10, 8))  # Create figure and axes explicitly

                    # Draw the tree on the created axes
                    Phylo.draw(biophylo_tree, axes=ax)
                    ax.set_title("UPGMA Phylogenetic Tree")
                    plt.show()  # Display the plot
                    print("Graphical tree displayed using Matplotlib.")
                except ImportError:
                    print(
                        "Matplotlib not installed. Skipping direct graphical display. Install with 'pip install matplotlib biopython'.")
                except Exception as e:
                    print(f"Error drawing with Matplotlib: {e}")

            except Exception as e:
                print(f"Error converting or saving graphical tree: {e}")
                print("Make sure Biopython is installed (`pip install biopython`).")
        else:
            print("UPGMA tree could not be constructed.")
    else:
        print("Program terminated due to invalid input data.")


if __name__ == "__main__":
    main()