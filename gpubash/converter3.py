import numpy as np
import ase
from ase import Atoms
from ase.geometry import wrap_positions
import math
VERBOSE = True


class Converter:
    """A coordinate converter class with lattice support using ASE"""

    def __init__(self):
        # Dictionary of the masses of elements indexed by element name
        self.masses = {'X': 0, 'Ac': 227.028, 'Al': 26.981539, 'Am': 243, 'Sb': 121.757, 'Ar': 39.948, 'As': 74.92159,
                       'At': 210, 'Ba': 137.327, 'Bk': 247, 'Be': 9.012182, 'Bi': 208.98037, 'Bh': 262, 'B': 10.811,
                       'Br': 79.904, 'Cd': 112.411, 'Ca': 40.078, 'Cf': 251, 'C': 12.011, 'Ce': 140.115,
                       'Cs': 132.90543, 'Cl': 35.4527, 'Cr': 51.9961, 'Co': 58.9332, 'Cu': 63.546, 'Cm': 247, 'Db': 262,
                       'Dy': 162.5, 'Es': 252, 'Er': 167.26, 'Eu': 151.965, 'Fm': 257, 'F': 18.9984032, 'Fr': 223,
                       'Gd': 157.25, 'Ga': 69.723, 'Ge': 72.61, 'Au': 196.96654, 'Hf': 178.49, 'Hs': 265,
                       'He': 4.002602, 'Ho': 164.93032, 'H': 1.00794, 'In': 114.82, 'I': 126.90447, 'Ir': 192.22,
                       'Fe': 55.847, 'Kr': 83.8, 'La': 138.9055, 'Lr': 262, 'Pb': 207.2, 'Li': 6.941, 'Lu': 174.967,
                       'Mg': 24.305, 'Mn': 54.93805,
                       'Mt': 266, 'Md': 258, 'Hg': 200.59, 'Mo': 95.94, 'Nd': 144.24, 'Ne': 20.1797, 'Np': 237.048,
                       'Ni': 58.6934, 'Nb': 92.90638, 'N': 14.00674, 'No': 259, 'Os': 190.2, 'O': 15.9994, 'Pd': 106.42,
                       'P': 30.973762, 'Pt': 195.08, 'Pu': 244, 'Po': 209, 'K': 39.0983, 'Pr': 140.90765, 'Pm': 145,
                       'Pa': 231.0359, 'Ra': 226.025, 'Rn': 222, 'Re': 186.207, 'Rh': 102.9055, 'Rb': 85.4678,
                       'Ru': 101.07, 'Rf': 261, 'Sm': 150.36, 'Sc': 44.95591, 'Sg': 263, 'Se': 78.96, 'Si': 28.0855,
                       'Ag': 107.8682, 'Na': 22.989768, 'Sr': 87.62, 'S': 32.066, 'Ta': 180.9479, 'Tc': 98, 'Te': 127.6,
                       'Tb': 158.92534, 'Tl': 204.3833, 'Th': 232.0381, 'Tm': 168.93421, 'Sn': 118.71, 'Ti': 47.88,
                       'W': 183.85, 'U': 238.0289, 'V': 50.9415, 'Xe': 131.29, 'Yb': 173.04, 'Y': 88.90585, 'Zn': 65.39,
                       'Zr': 91.224}
        self.total_mass = 0
        self.cartesian = []
        self.zmatrix = []
        self.lattice = None  # 存储晶格向量
        self.pbc = "T T T"  # 存储周期性边界条件
        self.original_atoms = None  # 存储原始ASE原子对象
        self.original_lattice = None  # 存储原始晶格信息
        self.original_frac_positions = None  # 存储原始分数坐标

    def read_zmatrix(self, input_file='zmatrix.dat'):
        """
        Read the input zmatrix file (assumes no errors and no variables)
        The zmatrix is a list with each element formatted as follows
        [ name, [[ atom1, distance ], [ atom2, angle ], [ atom3, dihedral ]], mass ]
        The first three atoms have blank lists for the undefined coordinates
        """
        self.zmatrix = []
        with open(input_file, 'r') as f:
            next(f)
            next(f)

            name = next(f).strip()
            self.zmatrix.append([name, [], self.masses[name]])
            name, atom1, distance = next(f).split()[:3]
            self.zmatrix.append([name,
                                 [[int(atom1) - 1, float(distance)], [], []],
                                 self.masses[name]])
            name, atom1, distance, atom2, angle = next(f).split()[:5]
            self.zmatrix.append([name,
                                 [[int(atom1) - 1, float(distance)],
                                  [int(atom2) - 1, np.radians(float(angle))], []],
                                 self.masses[name]])
            for line in f:
                # Get the components of each line, dropping anything extra
                name, atom1, distance, atom2, angle, atom3, dihedral = line.split()[:7]
                # convert to a base 0 indexing system and use radians
                atom = [name,
                        [[int(atom1) - 1, float(distance)],
                         [int(atom2) - 1, np.radians(float(angle))],
                         [int(atom3) - 1, np.radians(float(dihedral))]],
                        self.masses[name]]

                self.zmatrix.append(atom)

        return self.zmatrix

    def read_cartesian(self, input_file='cartesian.dat'):
        """
        Read the cartesian coordinates file with lattice information
        The cartesian coordiantes consist of a list of atoms formatted as follows
        [ name, np.array( [ x, y, z ] ), mass ]
        """
        self.cartesian = []
        with open(input_file, 'r') as f:
            # Read number of atoms
            num_atoms_line = next(f).strip()
            try:
                num_atoms = int(num_atoms_line)
            except ValueError:
                # If first line isn't a number, it might be a header
                num_atoms = int(next(f).strip())

            # Read lattice line
            lattice_line = next(f).strip()
            self._parse_lattice_line(lattice_line)
            self.original_lattice = self.lattice.copy() if self.lattice is not None else None

            # Read atoms
            positions = []
            symbols = []
            masses = []
            for i in range(num_atoms):
                line = next(f).split()
                if len(line) < 4:
                    continue  # Skip incomplete lines
                name = line[0]
                coords = list(map(float, line[1:4]))
                mass = self.masses.get(name, 0.0)  # Handle unknown elements
                self.cartesian.append([name, np.array(coords, dtype='f8'), mass])
                symbols.append(name)
                positions.append(coords)
                masses.append(mass)

            # 创建ASE原子对象
            pbc_flags = [flag == "T" for flag in self.pbc.split()]
            self.original_atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=self.lattice,
                pbc=pbc_flags
            )

            # 计算并存储分数坐标
            if self.lattice is not None:
                inv_lattice = np.linalg.inv(self.lattice)
                self.original_frac_positions = []
                for pos in positions:
                    frac = np.dot(inv_lattice, pos)
                    self.original_frac_positions.append(frac)

        return self.cartesian

    def _parse_lattice_line(self, line):
        """Parse the lattice information from a string"""
        # Reset lattice and pbc
        self.lattice = None
        self.pbc = "T T T"

        # Extract lattice information
        if 'Lattice=' in line:
            start = line.find('"') + 1
            end = line.find('"', start)
            if start > 0 and end > start:
                lattice_str = line[start:end]
                try:
                    # Convert to 9 floats
                    lattice_values = list(map(float, lattice_str.split()))
                    if len(lattice_values) == 9:
                        # Reshape to 3x3 matrix
                        self.lattice = np.array(lattice_values).reshape(3, 3)
                except ValueError:
                    pass  # Keep self.lattice as None

        # Extract PBC information
        if 'pbc=' in line:
            start = line.find('pbc="') + 5
            end = line.find('"', start)
            if start > 4 and end > start:
                self.pbc = line[start:end]

    def rotation_matrix(self, axis, angle):
        """
        Euler-Rodrigues formula for rotation matrix with improved numerical stability
        """
        # Check for zero axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            print('now return identity')
            return np.identity(3)  # Return identity matrix for zero axis

        # Normalize the axis
        axis = axis / axis_norm
        a = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)

        # Handle very small angles to avoid numerical issues
        if abs(sin_half) < 1e-10:
            print('now return small angles')
            b, c, d = 0, 0, 0
        else:
            b, c, d = -axis * sin_half

        return np.array([
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]
        ])

    def add_first_three_to_cartesian(self):
        """
        The first three atoms in the zmatrix need to be treated differently
        """
        # First atom
        name, coords, mass = self.zmatrix[0]
        self.cartesian = [[name, np.array([0., 0., 0.]), mass]]

        # Second atom
        name, coords, mass = self.zmatrix[1]
        distance = coords[0][1]
        self.cartesian.append(
            [name, np.array([distance, 0., 0.]), self.masses[name]])

        # Third atom
        name, coords, mass = self.zmatrix[2]
        atom1, atom2 = coords[:2]
        atom1, distance = atom1
        atom2, angle = atom2
        q = np.array(self.cartesian[atom1][1], dtype='f8')  # position of atom 1
        r = np.array(self.cartesian[atom2][1], dtype='f8')  # position of atom 2

        # Vector pointing from q to r
        a = r - q


        # Vector of length distance pointing along the x-axis
        d = distance * a / np.sqrt(np.dot(a, a))

        # Rotate d by the angle around the z-axis
        d = np.dot(self.rotation_matrix([0, 0, 1], angle), d)

        # Add d to the position of q to get the new coordinates of the atom
        p = q + d
        atom = [name, p, self.masses[name]]
        self.cartesian.append(atom)

    def add_atom_to_cartesian(self, coords):
        name, coords, mass = coords
        atom1, distance = coords[0]
        atom2, angle = coords[1]
        atom3, dihedral = coords[2]

        q = self.cartesian[atom1][1]  # atom 1
        r = self.cartesian[atom2][1]  # atom 2
        s = self.cartesian[atom3][1]  # atom 3

        # Calculate vector a (from q to r) and normalize
        a = r - q
        a_norm = np.linalg.norm(a)
        if a_norm < 1e-10:
            a = np.array([1.0, 0.0, 0.0])
        else:
            a = a / a_norm

        # Calculate vector b (from s to r) and normalize
        b = r - s
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-10:
            b = np.array([0.0, 1.0, 0.0])
        else:
            b = b / b_norm

        # Calculate normal vector to the plane defined by a and b
        normal = np.cross(a, b)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = normal / normal_norm

        # Calculate the local coordinate system
        # Assume a is along the z-axis, and normal is along the y-axis?
        # Alternatively, use trigonometry directly
        x = distance * math.sin(angle) * math.cos(dihedral)
        y = distance * math.sin(angle) * math.sin(dihedral)
        z = distance * math.cos(angle)
        offset = np.array([x, y, z])

        # Transform offset to the global coordinate system (simplified)
        # This part may need adjustment based on the actual reference frame
        # Here, we assume that the local frame is defined by a, normal, and cross(normal, a)
        local_z = a
        local_y = normal
        local_x = np.cross(local_y, local_z)

        # Project offset to global coordinates
        global_offset = offset[0] * local_x + offset[1] * local_y + offset[2] * local_z
        p = q + global_offset

        atom = [name, p, mass]
        self.cartesian.append(atom)

    def zmatrix_to_cartesian(self):
        """
        Convert the zmartix to cartesian coordinates
        """
        # Deal with first three line separately
        self.add_first_three_to_cartesian()

        for atom in self.zmatrix[3:]:
            self.add_atom_to_cartesian(atom)

        self.remove_dummy_atoms()
        self.center_cartesian()

        return self.cartesian

    def add_first_three_to_zmatrix(self):
        """The first three atoms need to be treated differently"""
        # First atom
        self.zmatrix = []
        name, position, mass = self.cartesian[0]
        self.zmatrix.append([name, [[], [], []], mass])

        # Second atom
        if len(self.cartesian) > 1:
            name, position, mass = self.cartesian[1]
            atom1 = self.cartesian[0]
            pos1 = atom1[1]
            q = pos1 - position
            distance = np.sqrt(np.dot(q, q))
            self.zmatrix.append([name, [[0, distance], [], []], mass])

        # Third atom
        if len(self.cartesian) > 2:
            name, position, mass = self.cartesian[2]
            atom1, atom2 = self.cartesian[:2]
            pos1, pos2 = atom1[1], atom2[1]
            q = pos1 - position
            r = pos2 - pos1
            q_u = q / np.sqrt(np.dot(q, q))
            r_u = r / np.sqrt(np.dot(r, r))
            distance = np.sqrt(np.dot(q, q))
            # Angle between a and b = acos(dot(a, b)) / (|a| |b|))
            angle = np.arccos(np.dot(-q_u, r_u))
            self.zmatrix.append(
                [name, [[0, distance], [1, np.degrees(angle)], []], mass])

    def add_atom_to_zmatrix(self, i, line):
        """Generates an atom for the zmatrix with improved numerical stability"""
        name, position, mass = line
        atom1, atom2, atom3 = self.cartesian[:3]
        pos1, pos2, pos3 = atom1[1], atom2[1], atom3[1]

        # 计算相关向量
        q = pos1 - position
        r = pos2 - pos1
        s = pos3 - pos2

        # 计算距离和角度（使用安全机制）
        distance = np.linalg.norm(q)

        # 角度计算（带安全保护）
        if np.linalg.norm(q) > 1e-10 and np.linalg.norm(r) > 1e-10:
            q_u = q / np.linalg.norm(q)
            r_u = r / np.linalg.norm(r)
            cos_angle = np.dot(-q_u, r_u)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
        else:
            # 如果向量太小，使用90度作为默认值
            angle = np.pi / 2

        # 二面角计算（带安全保护）
        plane1 = np.cross(q, r)
        plane2 = np.cross(r, s)

        norm1 = np.linalg.norm(plane1)
        norm2 = np.linalg.norm(plane2)

        if norm1 > 1e-10 and norm2 > 1e-10:
            dot_product = np.dot(plane1, plane2) / (norm1 * norm2)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            dihedral = np.arccos(dot_product)

            # 计算有符号二面角
            cross_product = np.cross(plane1, plane2)
            if np.linalg.norm(cross_product) > 1e-10:
                sign = np.sign(np.dot(cross_product, r))
                dihedral *= sign
        else:
            # 当平面无法定义时，使用默认值
            dihedral = 0.0

        coords = [[0, distance], [1, np.degrees(angle)], [2, np.degrees(dihedral)]]
        atom = [name, coords, mass]
        self.zmatrix.append(atom)

    def cartesian_to_zmatrix(self):
        """Convert the cartesian coordinates to a zmatrix"""
        self.add_first_three_to_zmatrix()
        for i, atom in enumerate(self.cartesian[3:], start=3):
            self.add_atom_to_zmatrix(i, atom)

        return self.zmatrix

    def remove_dummy_atoms(self):
        """Delete any dummy atoms that may have been placed in the calculated cartesian coordinates"""
        new_cartesian = []
        for atom, xyz, mass in self.cartesian:
            if not atom == 'X':
                new_cartesian.append([atom, xyz.copy(), mass])  # ✅ 改为列表
        self.cartesian = new_cartesian


    def center_cartesian(self):
        """Find the center of mass and move it to the origin"""
        self.total_mass = 0.0
        center_of_mass = np.array([0.0, 0.0, 0.0])
        for atom, xyz, mass in self.cartesian:
            self.total_mass += mass
            center_of_mass += xyz * mass
        center_of_mass = center_of_mass / self.total_mass

        # Translate each atom by the center of mass
        for atom, xyz, mass in self.cartesian:
            xyz -= center_of_mass

    def cartesian_radians_to_degrees(self):
        for atom in self.cartesian:
            atom[1][1][1] = np.degrees(atom[1][1][1])
            atom[1][2][1] = np.degrees(atom[1][2][1])

    def _format_lattice_string(self):
        """Format the lattice information as a string"""
        if self.lattice is None:
            return ""

        # Flatten the 3x3 lattice matrix
        lattice_flat = self.lattice.flatten()
        # Format with 9 decimal places
        lattice_str = " ".join([f"{val:.9f}" for val in lattice_flat])
        return f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3 pbc="{self.pbc}"'

    def output_cartesian(self, output_file='cartesian.xyz'):
        """Output the cartesian coordinates with lattice information"""
        with open(output_file, 'w') as f:
            # Write number of atoms
            f.write(f'{len(self.cartesian)}\n')

            # Write lattice information
            lattice_str = self._format_lattice_string()
            if lattice_str:
                f.write(f'{lattice_str}\n')
            else:
                f.write('\n')  # Empty line if no lattice

            # Write atomic coordinates
            f.write(self.str_cartesian())

    def str_cartesian(self):
        """Format the cartesian coordinates as a string"""
        out = ''
        for atom, (x, y, z), mass in self.cartesian:
            # Fixed width formatting for alignment
            out += f'{atom:<2s} {x:>15.9f} {y:>15.9f} {z:>15.9f}\n'
        return out

    def output_zmatrix(self, output_file='zmatrix.zmat'):
        """Output the zmatrix to the file"""
        with open(output_file, 'w') as f:
            f.write('#ZMATRIX\n#\n')
            f.write(self.str_zmatrix())

    def str_zmatrix(self):
        """Format the zmatrix as a string"""
        out = f'{self.zmatrix[0][0]}\n'
        for atom, position, mass in self.zmatrix[1:]:
            out += f'{atom:<2s}'
            for i in position:
                for j in range(0, len(i), 2):
                    out += f' {i[j] + 1:>3d} {i[j + 1]:>15.10f}'
            out += '\n'

        return out

    def match_original_cell_wrapping(self):
        """
        将新生成的原子位置匹配原始晶胞的包裹范围
        1. 计算原始结构的质心位置
        2. 计算新结构的质心位置
        3. 将新结构平移到原始质心位置
        4. 将新结构包裹到原始晶胞内
        """
        if self.original_atoms is None or self.lattice is None:
            return

        # 获取原始结构的质心位置
        original_com = self.original_atoms.get_center_of_mass()

        # 获取新结构的质心位置
        new_positions = np.array([atom[1] for atom in self.cartesian])
        new_com = np.mean(new_positions, axis=0)

        # 计算平移向量
        translation = original_com - new_com

        # 应用平移
        for atom in self.cartesian:
            atom[1] += translation

        # 使用原始晶格包裹原子位置
        if self.original_frac_positions is not None:
            # 使用原始分数坐标放置原子
            for i, atom in enumerate(self.cartesian):
                # 根据分数坐标和原始晶格计算笛卡尔坐标
                atom[1] = np.dot(self.original_lattice, self.original_frac_positions[i])
        else:
            # 如果没有原始分数坐标，则使用晶格包裹
            wrapped_positions = wrap_positions(
                positions=[atom[1] for atom in self.cartesian],
                cell=self.original_lattice,
                pbc=self.original_atoms.pbc
            )

            for i, atom in enumerate(self.cartesian):
                atom[1] = wrapped_positions[i]

        # 恢复原始晶格
        self.lattice = self.original_lattice.copy()

    def run_zmatrix(self, input_file='zmatrix.zmat', output_file='cartesian.xyz'):
        """Read in the zmatrix, converts it to cartesian, and outputs it to a file"""
        self.read_zmatrix(input_file)
        self.zmatrix_to_cartesian()  # 生成新的cartesian

        # 如果存在原始晶格信息，匹配原始晶胞的包裹范围
        if self.original_atoms is not None and self.original_lattice is not None:
            self.match_original_cell_wrapping()
        elif self.lattice is not None:
            # 如果没有原始晶格，但当前有晶格，则使用ASE调整晶格
            self.adjust_lattice_to_structure()

        self.output_cartesian(output_file)

    def run_cartesian(self, input_file='cartesian.xyz', output_file='zmatrix.zmat'):
        """Read in the cartesian coordinates, convert to zmatrix, and output the file"""
        self.read_cartesian(input_file)

        # 如果存在晶格信息，使用ASE调整晶格以匹配原子结构
        if self.lattice is not None:
            self.adjust_lattice_to_structure()

        self.cartesian_to_zmatrix()
        self.output_zmatrix(output_file)

    def set_lattice(self, lattice_matrix, pbc="T T T"):
        """Set the lattice manually"""
        if isinstance(lattice_matrix, list):
            lattice_matrix = np.array(lattice_matrix)
        if lattice_matrix.shape == (3, 3):
            self.lattice = lattice_matrix
            self.pbc = pbc
        else:
            raise ValueError("Lattice must be a 3x3 matrix")

    def get_lattice(self):
        """Get the current lattice"""
        return self.lattice


    # def adjust_lattice_to_structure(self):
    #     """使用ASE智能调整晶格和原子位置，确保原子间有合理距离"""
    #     if not self.cartesian:
    #         return
    #
    #     # 获取原子符号和位置
    #     symbols = [atom[0] for atom in self.cartesian]
    #     positions = np.array([atom[1] for atom in self.cartesian])
    #
    #     # 创建ASE原子对象
    #     pbc_flags = [flag == "T" for flag in self.pbc.split()]
    #     atoms = Atoms(
    #         symbols=symbols,
    #         positions=positions,
    #         cell=self.lattice if self.lattice is not None else np.eye(3),
    #         pbc=pbc_flags
    #     )
    #
    #     # 如果原始晶格存在，使用原始晶格方向但调整大小
    #     if self.original_lattice is not None:
    #         # 获取原始晶格向量和长度
    #         a, b, c = self.original_lattice
    #         a_len = np.linalg.norm(a)
    #         b_len = np.linalg.norm(b)
    #         c_len = np.linalg.norm(c)
    #
    #         # 获取原始晶格方向
    #         a_dir = a / a_len
    #         b_dir = b / b_len
    #         c_dir = c / c_len
    #
    #         # 计算原子在原始晶格方向上的投影范围
    #         min_a = min(np.dot(positions, a_dir))
    #         max_a = max(np.dot(positions, a_dir))
    #         min_b = min(np.dot(positions, b_dir))
    #         max_b = max(np.dot(positions, b_dir))
    #         min_c = min(np.dot(positions, c_dir))
    #         max_c = max(np.dot(positions, c_dir))
    #
    #         # 计算新晶格长度（添加10%缓冲）
    #         new_a_len = (max_a - min_a) * 1.10
    #         new_b_len = (max_b - min_b) * 1.10
    #         new_c_len = (max_c - min_c) * 1.10
    #
    #         # 创建新晶格（保持原始方向）
    #         new_cell = np.array([
    #             a_dir * new_a_len,
    #             b_dir * new_b_len,
    #             c_dir * new_c_len
    #         ])
    #
    #         # 设置新晶格
    #         atoms.set_cell(new_cell)
    #         atoms.wrap()  # 包裹原子到晶格内
    #
    #         # 计算晶格中心
    #         cell_center = np.sum(new_cell, axis=0) / 2
    #
    #         # 计算原子中心
    #         atom_center = atoms.get_center_of_mass()
    #
    #         # 平移原子使中心对齐
    #         atoms.translate(cell_center - atom_center)
    #     else:
    #         # 如果没有原始晶格，创建新的正交晶格
    #         min_pos = np.min(positions, axis=0)
    #         max_pos = np.max(positions, axis=0)
    #         size = max_pos - min_pos
    #
    #         # 计算缓冲大小（取最大尺寸的20%）
    #         buffer = np.max(size) * 0.20
    #
    #         # 应用缓冲
    #         min_pos -= buffer
    #         max_pos += buffer
    #         size = max_pos - min_pos
    #
    #         # 创建新的正交晶格
    #         new_cell = np.diag(size)
    #         atoms.set_cell(new_cell)
    #         atoms.center()  # 将原子居中在晶格内
    #
    #     # 更新类中的晶格和原子位置
    #     self.lattice = atoms.get_cell().array
    #     for i, atom in enumerate(self.cartesian):
    #         atom[1] = atoms.positions[i]
    #
    #     # 更新PBC设置
    #     self.pbc = " ".join(["T" if flag else "F" for flag in atoms.pbc])
    #
    #     # 更新分数坐标
    #     inv_lattice = np.linalg.inv(self.lattice)
    #     self.original_frac_positions = [np.dot(inv_lattice, pos) for pos in atoms.positions]
    #
    #     # 更新原始晶格信息（如果不存在）
    #     if self.original_lattice is None:
    #         self.original_lattice = self.lattice.copy()
    #         self.original_atoms = atoms

    def adjust_lattice_to_structure(self):
        """根据结构类型智能调整晶格：表面结构保持真空层，体相结构调整尺寸"""
        if not self.cartesian or self.lattice is None:
            return

        # 获取原子符号和位置
        symbols = [atom[0] for atom in self.cartesian]
        positions = np.array([atom[1] for atom in self.cartesian])

        # 创建ASE原子对象
        pbc_flags = [flag == "T" for flag in self.pbc.split()]
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=self.lattice,
            pbc=pbc_flags
        )

        # 计算结构在晶胞中的填充密度
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        atom_span = max_pos - min_pos

        # 计算晶胞尺寸
        cell_lengths = np.array([
            np.linalg.norm(self.lattice[0]),
            np.linalg.norm(self.lattice[1]),
            np.linalg.norm(self.lattice[2])
        ])

        # 计算每个方向的填充比例
        fill_ratios = atom_span / cell_lengths

        # 判断结构类型
        is_surface = np.any(fill_ratios < 0.8)  # 任一方向填充率低于80%视为表面结构
        is_bulk = np.all(fill_ratios > 0.9)  # 所有方向填充率高于90%视为体相结构

        # 根据不同结构类型采取不同调整策略
        if is_surface:
            # 表面结构：保持晶胞尺寸，只调整原子位置
            self._adjust_surface_structure(atoms, fill_ratios)
        elif is_bulk:
            # 体相结构：调整晶胞尺寸以包裹原子
            self._adjust_bulk_structure(atoms)
        else:
            # 中间状态：保持原始晶胞方向但调整尺寸
            self._adjust_general_structure(atoms, fill_ratios)

        # 更新类中的晶格和原子位置
        self.lattice = atoms.get_cell().array
        for i, atom in enumerate(self.cartesian):
            atom[1] = atoms.positions[i]

        # 更新PBC设置
        self.pbc = " ".join(["T" if flag else "F" for flag in atoms.pbc])

        # 更新分数坐标
        inv_lattice = np.linalg.inv(self.lattice)
        self.original_frac_positions = [np.dot(inv_lattice, pos) for pos in atoms.positions]

        # 更新原始晶格信息
        self.original_lattice = self.lattice.copy()
        self.original_atoms = atoms

    def _adjust_surface_structure(self, atoms, fill_ratios):
        """调整表面结构：保持晶胞尺寸，优化原子位置"""
        # 识别有真空层的方向
        vacuum_directions = [i for i, ratio in enumerate(fill_ratios) if ratio < 0.8]

        # 在这些方向上将原子居中
        if vacuum_directions:
            # 计算原子中心
            center = atoms.get_center_of_mass()

            # 计算晶胞中心
            cell_center = np.sum(atoms.get_cell(), axis=0) / 2

            # 计算平移向量（只在有真空层的方向平移）
            translation = np.zeros(3)
            for dim in vacuum_directions:
                translation[dim] = cell_center[dim] - center[dim]

            # 应用平移
            atoms.translate(translation)

        # 包裹原子到晶格内
        atoms.wrap()

    def _adjust_bulk_structure(self, atoms):
        """调整体相结构：按比例缩放晶胞以包裹原子"""
        # 计算原子在晶胞方向上的投影范围
        min_proj = []
        max_proj = []
        for i in range(3):
            axis = atoms.get_cell()[i] / np.linalg.norm(atoms.get_cell()[i])
            projections = np.dot(atoms.positions, axis)
            min_proj.append(np.min(projections))
            max_proj.append(np.max(projections))

        # 计算新晶胞尺寸（添加5%缓冲）
        new_lengths = [(max_p - min_p) * 1.05 for min_p, max_p in zip(min_proj, max_proj)]

        # 保持原始晶胞方向，只调整尺寸
        new_cell = atoms.get_cell()
        for i in range(3):
            norm = np.linalg.norm(new_cell[i])
            new_cell[i] = new_cell[i] / norm * new_lengths[i]

        atoms.set_cell(new_cell)
        atoms.wrap()

        # 将原子居中在新晶胞内
        atoms.center()

    def _adjust_general_structure(self, atoms, fill_ratios):
        """调整一般结构：保持原始方向，按需调整尺寸"""
        # 计算需要调整的方向
        adjust_directions = [i for i, ratio in enumerate(fill_ratios) if ratio > 0.9]

        # 计算原子在晶胞方向上的投影范围
        min_proj = []
        max_proj = []
        for i in range(3):
            axis = atoms.get_cell()[i] / np.linalg.norm(atoms.get_cell()[i])
            projections = np.dot(atoms.positions, axis)
            min_proj.append(np.min(projections))
            max_proj.append(np.max(projections))

        # 创建新晶胞（保持原始方向）
        new_cell = atoms.get_cell().copy()
        for i in range(3):
            if i in adjust_directions:
                # 对于填充率高的方向，添加5%缓冲
                new_length = (max_proj[i] - min_proj[i]) * 1.05
                norm = np.linalg.norm(new_cell[i])
                new_cell[i] = new_cell[i] / norm * new_length

        atoms.set_cell(new_cell)
        atoms.wrap()

        # 将原子居中
        atoms.center()

