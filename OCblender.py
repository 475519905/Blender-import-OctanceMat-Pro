import bpy
import os
import json
from bpy.types import Operator, AddonPreferences


bl_info = {
    "name": "Import Octane Material Pro",
    "blender": (4, 2, 0),  # Adjust according to your Blender version
    "category": "Import-Export",
    "author": "475519905",
    "version": (2, 1, 2),
    "description": "Imports and applies Octane materials directly into Blender materials panel.",
}

def vector_to_dict(vec):
    """Converts a vector dictionary to one containing x, y, z keys."""
    if isinstance(vec, dict):
        return {'x': vec.get('x', 0.0), 'y': vec.get('y', 0.0), 'z': vec.get('z', 0.0)}
    elif isinstance(vec, (list, tuple)) and len(vec) >= 3:
        return {'x': vec[0], 'y': vec[1], 'z': vec[2]}
    else:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}

def parse_material_info(file_path):
    """Parses the material information file and returns materials and shader mapping."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return {}, {}
    
    materials = {}
    shader_info_by_id = {}
    for item in data:
        material_info = item.get('material_info', {})
        material_name = material_info.get('material_name', 'Unnamed_Material')
        materials[material_name] = material_info
        # Collect shaders from the material
        collect_shaders(material_info, shader_info_by_id)
    return materials, shader_info_by_id

def collect_shaders(shader_info, shader_info_by_id):
    """Recursively collects shaders and their unique IDs."""
    if not isinstance(shader_info, dict):
        return
    unique_id = shader_info.get('unique_id')
    if unique_id:
        shader_info_by_id[unique_id] = shader_info
    # Recursively collect shaders from all dictionary values
    for key, value in shader_info.items():
        if isinstance(value, dict):
            collect_shaders(value, shader_info_by_id)
        elif isinstance(value, list):
            for item in value:
                collect_shaders(item, shader_info_by_id)

def create_texture_node(nodes, links, y_offset=0):
    """Creates an image texture node with mapping node."""
    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.location = (-300, y_offset)
    
    # Create mapping node
    mapping_node = nodes.new(type='ShaderNodeMapping')
    mapping_node.location = (-600, y_offset)
    
    # Create texture coordinate node if it doesn't exist
    tex_coord_node = nodes.get('Texture Coordinate')
    if not tex_coord_node:
        tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
        tex_coord_node.location = (-900, y_offset)
    
    # Connect: Texture Coordinate → Mapping → Image Texture
    links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
    
    return tex_node

def create_hue_sat_node(nodes, links, y_offset=0):
    """Creates a Hue/Saturation node."""
    hue_sat_node = nodes.new(type='ShaderNodeHueSaturation')
    hue_sat_node.location = (-300, y_offset)
    return hue_sat_node

def create_rgb_node(nodes, links, color, y_offset=0):
    """Creates an RGB node."""
    color_node = nodes.new(type='ShaderNodeRGB')
    color_node.location = (-600, y_offset)
    color_node.outputs['Color'].default_value = (color['x'], color['y'], color['z'], 1.0)
    return color_node

def create_math_node(nodes, links, operation, value, y_offset=0):
    """Creates a Math node."""
    math_node = nodes.new(type='ShaderNodeMath')
    math_node.location = (-300, y_offset)
    math_node.operation = operation
    math_node.inputs[1].default_value = value
    return math_node

def create_value_node(nodes, links, value, y_offset=0):
    """Creates a Value node."""
    value_node = nodes.new(type='ShaderNodeValue')
    value_node.location = (-600, y_offset)
    value_node.outputs[0].default_value = value
    return value_node


def process_noise(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a Noise shader by creating an Image Texture node in Blender.
    """
    noise_shader_file = shader_info.get('preview_image')
    if not noise_shader_file:
        print("Warning: Noise shader is missing 'preview_image'")
        return None

    # 调整图像路径
    image_path = noise_shader_file
    if not os.path.isabs(image_path):
        image_path = os.path.join(base_path, image_path)
    else:
        # 使用 os.path.normpath 规范化路径，处理反斜杠问题
        image_path = os.path.normpath(image_path)

    if not os.path.exists(image_path):
        print(f"Warning: Noise image file not found: {image_path}")
        return None

    try:
        # 加载图像
        img = bpy.data.images.load(image_path)
        
        # 创建 Image Texture 节点
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.location = (-300, y_offset)
        tex_node.image = img
        
        # 根据 input_name 设置颜色空间
        if input_name in ['Base Color', 'Emission']:
            tex_node.image.colorspace_settings.name = 'sRGB'
        else:
            tex_node.image.colorspace_settings.name = 'Non-Color'
        
        # 创建映射节点
        mapping_node = nodes.new(type='ShaderNodeMapping')
        mapping_node.location = (-600, y_offset)
        
        # 确保存在 Texture Coordinate 节点
        tex_coord_node = nodes.get('Texture Coordinate')
        if not tex_coord_node:
            tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
            tex_coord_node.location = (-900, y_offset)
        
        # 连接: Texture Coordinate → Mapping → Image Texture
        links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
        
        return tex_node.outputs['Color']
    
    except Exception as e:
        print(f"Error loading noise image {image_path}: {e}")
        return None


def process_shader(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Recursively processes a shader, resolving references using unique IDs.
    """
    # Resolve shader_info if it's a unique ID (string)
    if isinstance(shader_info, str):
        shader_info_resolved = shader_info_by_id.get(shader_info)
        if shader_info_resolved:
            shader_info = shader_info_resolved
        else:
            print(f"Error: Could not resolve shader_info with unique_id '{shader_info}'")
            return None

    if not isinstance(shader_info, dict):
        print(f"Error: shader_info is not a dictionary: {shader_info}")
        return None

    shader_type = shader_info.get('type')
    if shader_type in shader_processors:
        processor = shader_processors[shader_type]
        return processor(nodes, links, principled, shader_info, shader_info_by_id, y_offset, input_name, base_path)
    else:
        print(f"Unhandled shader type: {shader_type}")
        return None

def process_color_correction(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a ColorCorrection shader, applies hue/saturation adjustments,
    and returns the final color output socket.
    """
    color_correction_info = shader_info.get('color_correction_link', {})
    if not color_correction_info:
        print("Warning: ColorCorrection shader has no 'color_correction_link'")
        return None

    # Recursively process the linked shader to get the color output
    linked_shader_output = process_shader(nodes, links, principled, color_correction_info, shader_info_by_id, y_offset - 200, input_name, base_path)
    if not linked_shader_output:
        print("Warning: Failed to process linked shader in ColorCorrection")
        return None

    # Create Hue/Saturation node
    hue_sat_node = nodes.new(type='ShaderNodeHueSaturation')
    hue_sat_node.location = (-300, y_offset)

    # Set parameters (use default values or get from shader_info)
    hue = shader_info.get('hue', 0.5)
    saturation = shader_info.get('saturation', 1.0)
    value = shader_info.get('value', 1.0)
    fac = shader_info.get('fac', 1.0)

    hue_sat_node.inputs['Hue'].default_value = hue
    hue_sat_node.inputs['Saturation'].default_value = saturation
    hue_sat_node.inputs['Value'].default_value = value
    hue_sat_node.inputs['Fac'].default_value = fac

    # Connect the linked shader output to the Hue/Saturation node's Color input
    links.new(linked_shader_output, hue_sat_node.inputs['Color'])

    # Return the Hue/Saturation node's Color output
    return hue_sat_node.outputs['Color']

def process_gradient_shaders(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a Gradient shader by creating a ColorRamp node.
    """
    gradient_info = shader_info.get('gradient', {})
    knots = gradient_info.get('knots', [])
    if not knots:
        print("Warning: Gradient shader has no knots")
        return None

    # Create a ColorRamp node
    color_ramp_node = nodes.new(type='ShaderNodeValToRGB')
    color_ramp_node.location = (-300, y_offset)

    # Clear default elements but keep at least one
    elements = color_ramp_node.color_ramp.elements
    while len(elements) > 1:
        elements.remove(elements[-1])

    # Modify existing elements and add new ones as needed
    for i, knot in enumerate(knots):
        position = knot.get('position', 0.0)
        color_data = knot.get('color', {})
        color = (
            color_data.get('x', 0.0),
            color_data.get('y', 0.0),
            color_data.get('z', 0.0),
            1.0
        )
        if i < len(elements):
            # Modify existing element
            element = elements[i]
            element.position = position
            element.color = color
        else:
            # Add new element
            element = elements.new(position)
            element.color = color

    # Process gradient_texture_link if it exists
    gradient_texture_link = shader_info.get('gradient_texture_link')
    if gradient_texture_link:
        # Recursively process the linked shader
        linked_shader_output = process_shader(
            nodes, links, principled, gradient_texture_link, shader_info_by_id, y_offset - 200, input_name, base_path
        )
        if linked_shader_output:
            links.new(linked_shader_output, color_ramp_node.inputs['Fac'])
        else:
            print("Warning: Failed to process gradient_texture_link in Gradient shader")
    else:
        # If no link, connect a default value
        value_node = create_value_node(nodes, links, 0.5, y_offset - 200)
        links.new(value_node.outputs[0], color_ramp_node.inputs['Fac'])

    return color_ramp_node.outputs['Color']


def process_bitmap(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a Bitmap shader by creating an Image Texture node and applying power scaling.
    """
    bitmap_info = shader_info.get('invert', {}).get('bitmap', {})  # Assuming 'invert' key holds 'bitmap'; adjust if different
    bitmap_info = shader_info.get('invert', shader_info.get('bitmap', {}))  # Handle both cases
    
    bitmap_shader_file = shader_info.get('bitmap_shader_file')  # Adjust according to actual JSON structure
    power_float = shader_info.get('power_float', 1.0)
    power_link_info = shader_info.get('power_link')  # Currently null in JSON, handle if needed
    transform_link_info = shader_info.get('transform_link')  # Currently null in JSON, handle if needed
    
    if not bitmap_shader_file:
        print("Warning: Bitmap shader is missing 'bitmap_shader_file'")
        return None
    
    # Adjust the image path
    image_path = bitmap_shader_file
    if not os.path.isabs(image_path):
        image_path = os.path.join(base_path, image_path)
    else:
        # Normalize path to handle backslashes
        image_path = os.path.normpath(image_path)
    
    if not os.path.exists(image_path):
        print(f"Warning: Bitmap image file not found: {image_path}")
        return None
    
    try:
        # Load the image
        img = bpy.data.images.load(image_path)
        
        # Create Image Texture node
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.location = (-300, y_offset)
        tex_node.image = img
        
        # Set color space based on input_name
        if input_name in ['Base Color', 'Emission']:
            tex_node.image.colorspace_settings.name = 'sRGB'
        else:
            tex_node.image.colorspace_settings.name = 'Non-Color'
        
        # 创建映射节点
        mapping_node = nodes.new(type='ShaderNodeMapping')
        mapping_node.location = (-600, y_offset)
        
        # Add a Texture Coordinate node if needed
        tex_coord_node = nodes.get('Texture Coordinate')
        if not tex_coord_node:
            tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
            tex_coord_node.location = (-900, y_offset)
        
        # 连接: Texture Coordinate → Mapping → Image Texture
        links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
        
        # If power_float is not 1.0, apply a Math node to scale the texture
        if power_float != 1.0:
            math_node = nodes.new(type='ShaderNodeMath')
            math_node.operation = 'MULTIPLY'
            math_node.location = (-100, y_offset)
            math_node.inputs[1].default_value = power_float
            links.new(tex_node.outputs['Color'], math_node.inputs[0])
            return math_node.outputs['Value']
        else:
            return tex_node.outputs['Color']
    
    except Exception as e:
        print(f"Error loading bitmap image {image_path}: {e}")
        return None



def process_image_texture(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes an ImageTexture shader, creates a texture node, and returns its output socket.
    """
    image_path = shader_info.get('image_texture_file')
    if not image_path:
        print("Warning: ImageTexture shader has no 'image_texture_file'")
        return None

    # Adjust the image path
    if not os.path.isabs(image_path):
        image_path = os.path.join(base_path, image_path)
    else:
        # Replace the base path if needed (adjust according to your environment)
        # For example, replace 'C:\\Users\\475519905\\' with your actual base path
        base_dir = os.path.dirname(base_path)
        image_path = image_path.replace('C:\\Users\\475519906\\', base_dir + '/')

    if os.path.exists(image_path):
        try:
            img = bpy.data.images.load(image_path)
            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.location = (-300, y_offset)
            tex_node.image = img
            
            # Set color space based on input_name
            if input_name in ['Base Color', 'Emission']:
                tex_node.image.colorspace_settings.name = 'sRGB'
            else:
                tex_node.image.colorspace_settings.name = 'Non-Color'
            
            # 创建映射节点
            mapping_node = nodes.new(type='ShaderNodeMapping')
            mapping_node.location = (-600, y_offset)
            
            # Add a Texture Coordinate node if needed
            tex_coord_node = nodes.get('Texture Coordinate')
            if not tex_coord_node:
                tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
                tex_coord_node.location = (-900, y_offset)
            
            # 连接: Texture Coordinate → Mapping → Image Texture
            links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
            links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
            
            return tex_node.outputs['Color']
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    else:
        print(f"Warning: Image texture file not found: {image_path}")
        return None

def process_float_texture(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a FloatTexture shader, creates a Value node, and returns its output socket.
    Handles different key names for float values to ensure compatibility.
    """
    # 尝试从不同的键名获取浮点值，以兼容不同的 JSON 格式
    float_value = shader_info.get('float_texture_value') or shader_info.get('float_value', 1.0)
    
    # 创建 Value 节点
    value_node = create_value_node(nodes, links, float_value, y_offset)
    
    # 返回 Value 节点的输出
    return value_node.outputs[0]


def process_color_shader(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a color shader (including RgbSpectrum), creates an RGB node, and returns its Color output socket.
    """
    # Support 'color' and 'rgb_spectrum_color' fields
    color_data = shader_info.get('color') or shader_info.get('rgb_spectrum_color', {})
    if not color_data:
        print("Warning: No color data found in shader_info")
        return None
    color = vector_to_dict(color_data)

    # Create RGB node
    color_node = create_rgb_node(nodes, links, color, y_offset)

    # Return the RGB node's Color output
    return color_node.outputs['Color']

def process_bump_map(nodes, links, shader_output, y_offset=0):
    """
    Processes a Bump map and connects it through a Bump node.
    """
    bump_node = nodes.new(type='ShaderNodeBump')
    bump_node.location = (-100, y_offset)
    links.new(shader_output, bump_node.inputs['Height'])
    return bump_node.outputs['Normal']

def process_normal_map(nodes, links, shader_output, y_offset=0):
    """
    Processes a Normal map and connects it through a Normal Map node.
    """
    normal_map_node = nodes.new(type='ShaderNodeNormalMap')
    normal_map_node.location = (-100, y_offset)
    links.new(shader_output, normal_map_node.inputs['Color'])
    return normal_map_node.outputs['Normal']


def process_invert(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes an Invert shader by creating an Invert node in Blender.
    """
    invert_info = shader_info.get('invert', {})
    linked_texture_info = invert_info.get('linked_texture')

    if not linked_texture_info:
        print("Warning: Invert shader is missing 'linked_texture'")
        return None

    # Recursively process the linked texture
    linked_output = process_shader(
        nodes, links, principled, linked_texture_info, shader_info_by_id, y_offset - 200, None, base_path
    )

    if not linked_output:
        print("Warning: Invert shader failed to process 'linked_texture'")
        return None

    # Create Invert node
    invert_node = nodes.new(type='ShaderNodeInvert')
    invert_node.location = (-300, y_offset)

    # Connect the linked texture output to the Invert node
    links.new(linked_output, invert_node.inputs['Color'])

    # Return the Invert node's output
    return invert_node.outputs['Color']



def process_falloffmap(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a Falloffmap shader by creating a Fresnel node in Blender.
    Converts 'normal' to a vector if it's a single float.
    """
    falloffmap_info = shader_info.get('falloffmap', {})
    normal = falloffmap_info.get('normal', 0.0)
    grazing = falloffmap_info.get('grazing', 1.0)
    
    # 创建 Fresnel 节点
    fresnel_node = nodes.new(type='ShaderNodeFresnel')
    fresnel_node.location = (-600, y_offset)
    fresnel_node.inputs['IOR'].default_value = grazing  # Grazing 作为 IOR 使用
    
    # 处理 Normal 输入
    if isinstance(normal, (list, tuple)) and len(normal) == 3:
        # 如果 normal 是一个向量，直接赋值
        fresnel_node.inputs['Normal'].default_value = tuple(normal)
    elif isinstance(normal, (int, float)):
        # 如果 normal 是一个浮点数，将其转换为向量
        fresnel_node.inputs['Normal'].default_value = (0.0, 0.0, normal)
    else:
        print(f"Warning: Invalid 'normal' value in Falloffmap: {normal}")
        fresnel_node.inputs['Normal'].default_value = (0.0, 0.0, 1.0)  # 默认向量
    
    # 返回 Fresnel 节点的输出
    return fresnel_node.outputs['Fac']

def process_multiply(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a Multiply shader by creating a MixRGB node set to Multiply.
    """
    multiply_info = shader_info.get('multiply', {})
    texture1_info = multiply_info.get('texture1')
    texture2_info = multiply_info.get('texture2')
    
    if not texture1_info or not texture2_info:
        print("Warning: Multiply shader is missing texture1 or texture2")
        return None
    
    # Recursively process texture1 and texture2
    texture1_output = process_shader(nodes, links, principled, texture1_info, shader_info_by_id, y_offset - 200, None, base_path)
    texture2_output = process_shader(nodes, links, principled, texture2_info, shader_info_by_id, y_offset - 400, None, base_path)
    
    if not texture1_output or not texture2_output:
        print("Warning: Multiply shader failed to process texture1 or texture2")
        return None
    
    # Create MixRGB node set to Multiply
    mix_node = nodes.new(type='ShaderNodeMixRGB')
    mix_node.blend_type = 'MULTIPLY'
    mix_node.location = (-300, y_offset)
    links.new(texture1_output, mix_node.inputs['Color1'])
    links.new(texture2_output, mix_node.inputs['Color2'])
    
    return mix_node.outputs['Color']

def process_add(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes an Add shader by creating a MixRGB node set to Add.
    """
    add_info = shader_info.get('add', {})
    texture1_info = add_info.get('texture1')
    texture2_info = add_info.get('texture2')
    
    if not texture1_info or not texture2_info:
        print("Warning: Add shader is missing texture1 or texture2")
        return None
    
    # Recursively process texture1 and texture2
    texture1_output = process_shader(nodes, links, principled, texture1_info, shader_info_by_id, y_offset - 200, None, base_path)
    texture2_output = process_shader(nodes, links, principled, texture2_info, shader_info_by_id, y_offset - 400, None, base_path)
    
    if not texture1_output or not texture2_output:
        print("Warning: Add shader failed to process texture1 or texture2")
        return None
    
    # Create MixRGB node set to Add
    mix_node = nodes.new(type='ShaderNodeMixRGB')
    mix_node.blend_type = 'ADD'
    mix_node.location = (-300, y_offset)
    links.new(texture1_output, mix_node.inputs['Color1'])
    links.new(texture2_output, mix_node.inputs['Color2'])
    
    return mix_node.outputs['Color']




def process_mix_texture(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a MixTexture shader by blending two textures or colors.
    """
    mix_info = shader_info.get('mix_texture', {})
    mix_node = nodes.new(type='ShaderNodeMixRGB')
    mix_node.location = (-300, y_offset)
    
    # Set the mix factor
    amount_float = mix_info.get('amount_float', 0.5)
    mix_node.inputs['Fac'].default_value = amount_float
    
    # Process amount_link if it exists
    amount_link_info = mix_info.get('amount_link')
    if amount_link_info:
        amount_shader_output = process_shader(nodes, links, principled, amount_link_info, shader_info_by_id, y_offset - 200, None, base_path)
        if amount_shader_output:
            links.new(amount_shader_output, mix_node.inputs['Fac'])
    
    # Process texture1_link
    texture1_link_info = mix_info.get('texture1_link')
    if texture1_link_info:
        texture1_output = process_shader(nodes, links, principled, texture1_link_info, shader_info_by_id, y_offset - 400, None, base_path)
        if texture1_output:
            links.new(texture1_output, mix_node.inputs['Color1'])
    else:
        # If texture1_link is missing, use a default color
        mix_node.inputs['Color1'].default_value = (1.0, 1.0, 1.0, 1.0)
    
    # Process texture2_link
    texture2_link_info = mix_info.get('texture2_link')
    if texture2_link_info:
        texture2_output = process_shader(nodes, links, principled, texture2_link_info, shader_info_by_id, y_offset - 600, None, base_path)
        if texture2_output:
            links.new(texture2_output, mix_node.inputs['Color2'])
    else:
        # If texture2_link is missing, use a default color
        mix_node.inputs['Color2'].default_value = (0.0, 0.0, 0.0, 1.0)
    
    return mix_node.outputs['Color']

def process_gradient_shader(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a Gradient shader by creating a texture node using the gradient image.
    """
    gradient_info = shader_info.get('gradient', {})
    gradient_image_path = gradient_info.get('image_path')
    if not gradient_image_path or not os.path.exists(gradient_image_path):
        print("Warning: Gradient shader has no valid 'image_path'")
        return None

    # Create a texture node using the gradient image
    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.location = (-300, y_offset)
    img = bpy.data.images.load(gradient_image_path)
    tex_node.image = img
    
    # Set color space based on input_name
    if input_name in ['Base Color', 'Emission']:
        tex_node.image.colorspace_settings.name = 'sRGB'
    else:
        tex_node.image.colorspace_settings.name = 'Non-Color'
    
    # 创建映射节点
    mapping_node = nodes.new(type='ShaderNodeMapping')
    mapping_node.location = (-600, y_offset)
    
    # Add a Texture Coordinate node if needed
    tex_coord_node = nodes.get('Texture Coordinate')
    if not tex_coord_node:
        tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
        tex_coord_node.location = (-900, y_offset)
    
    # 连接: Texture Coordinate → Mapping → Image Texture
    links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
    
    return tex_node.outputs['Color']

def process_blackbody_emission(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a Blackbody Emission shader by processing its linked texture/color.
    """
    blackbody_link = shader_info.get('blackbody_link')
    if not blackbody_link:
        print("Warning: Blackbody Emission shader has no 'blackbody_link'")
        return None

    # Process the linked shader (usually an ImageTexture or Color)
    shader_output = process_shader(nodes, links, principled, blackbody_link, shader_info_by_id, y_offset, input_name, base_path)
    
    if shader_output:
        # Create a RGB to BW node to convert color to temperature
        rgb_to_bw = nodes.new(type='ShaderNodeRGBToBW')
        rgb_to_bw.location = (-300, y_offset)
        links.new(shader_output, rgb_to_bw.inputs['Color'])
        
        # Create Blackbody node
        blackbody = nodes.new(type='ShaderNodeBlackbody')
        blackbody.location = (-150, y_offset)
        links.new(rgb_to_bw.outputs['Val'], blackbody.inputs['Temperature'])
        
        return blackbody.outputs['Color']
    
    return None

def process_displacement(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    处理置换着色器,创建置换节点并连接到材质输出。
    """
    displacement_info = shader_info.get('displacement', {})
    if not displacement_info:
        print("警告: 置换着色器缺少 'displacement' 信息")
        return None

    # 获取置换量和纹理信息
    amount = displacement_info.get('amount', 1.0)
    texture_info = displacement_info.get('texture')
    
    if not texture_info:
        print("警告: 置换着色器缺少纹理信息")
        return None

    # 处理纹理
    texture_output = process_shader(nodes, links, principled, texture_info, shader_info_by_id, y_offset - 200, None, base_path)
    
    if texture_output:
        # 创建位移节点
        displacement_node = nodes.new(type='ShaderNodeDisplacement')
        displacement_node.location = (-150, y_offset)
        
        # 设置位移量
        displacement_node.inputs['Scale'].default_value = amount
        
        # 连接纹理到位移节点
        links.new(texture_output, displacement_node.inputs['Height'])
        
        return displacement_node.outputs['Displacement']
    
    return None

def process_texture_emission(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    Processes a Texture Emission shader by directly connecting the texture to emission.
    """
    texture_link = shader_info.get('texture_link')
    if not texture_link:
        print("Warning: Texture Emission shader has no 'texture_link'")
        return None

    # Process the linked shader (usually an ImageTexture)
    shader_output = process_shader(nodes, links, principled, texture_link, shader_info_by_id, y_offset, input_name, base_path)
    
    if shader_output:
        # 直接连接到 Emission 输入
        links.new(shader_output, principled.inputs['Emission Color'])
        principled.inputs['Emission Strength'].default_value = 1.0
        
        return shader_output
    
    return None

def process_layer(nodes, links, principled, shader_info, shader_info_by_id, y_offset=0, input_name=None, base_path=""):
    """
    处理图层着色器，使用预览图像创建纹理节点。
    """
    preview_image = shader_info.get('preview_image')
    if not preview_image:
        print("警告: 图层着色器缺少预览图像")
        return None

    # 调整图像路径 - 首先尝试原始路径
    image_path = preview_image
    if not os.path.isabs(image_path):
        image_path = os.path.join(base_path, image_path)
    else:
        image_path = os.path.normpath(image_path)
    
    # 如果原始路径不存在，尝试重构路径
    if not os.path.exists(image_path):
        if os.path.isabs(preview_image):
            # 尝试重构文件名
            dir_path = os.path.dirname(preview_image)
            light_name = os.path.basename(preview_image).split('_')[0]
            alt_image_path = os.path.join(dir_path, f"{light_name}_Efficiency_layer_preview.png")
            alt_image_path = os.path.normpath(alt_image_path)
            if os.path.exists(alt_image_path):
                image_path = alt_image_path
            else:
                print(f"警告: 图层预览图像未找到: {image_path}")
                print(f"也尝试了: {alt_image_path}")
                return None
        else:
            print(f"警告: 图层预览图像未找到: {image_path}")
            return None

    try:
        # 加载图像
        img = bpy.data.images.load(image_path)
        
        # 创建图像纹理节点
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.location = (-300, y_offset)
        tex_node.image = img
        
        # 根据input_name设置颜色空间
        if input_name in ['Base Color', 'Emission', 'Color']:
            tex_node.image.colorspace_settings.name = 'sRGB'
        else:
            tex_node.image.colorspace_settings.name = 'Non-Color'
        
        # 创建映射节点
        mapping_node = nodes.new(type='ShaderNodeMapping')
        mapping_node.location = (-600, y_offset)
        
        # 确保存在Texture Coordinate节点
        tex_coord_node = nodes.get('Texture Coordinate')
        if not tex_coord_node:
            tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
            tex_coord_node.location = (-900, y_offset)
        
        # 连接: Texture Coordinate → Mapping → Image Texture
        links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
        
        print(f"成功创建图层着色器纹理节点: {image_path}")
        return tex_node.outputs['Color']
    
    except Exception as e:
        print(f"加载图层预览图像时出错 {image_path}: {e}")
        return None


# Shader processing function dictionary
shader_processors = {
    1029512: process_color_correction,    # ColorCorrection
    1029508: process_image_texture,       # ImageTexture
    1029506: process_float_texture,       # FloatTexture
    5832: process_color_shader,           # Color Shader
    1029504: process_color_shader,        # RgbSpectrum (Color Shader)
    1029505: process_mix_texture,         # MixTexture
    1011100: process_gradient_shader,     # Gradient Shader
    1029513: process_gradient_shaders,    # Gradient Shader (newly added)
    1029503: process_falloffmap,  
    1029516: process_multiply,
    1038877: process_add, 
    1029514: process_invert,   
    5833: process_bitmap, 
    1011116: process_noise,  
    1029520: process_blackbody_emission,
    1031901: process_displacement,
    1029642: process_texture_emission,
    1011123: process_layer,
    # Add more shader types as needed
}

def process_mix_material(nodes, links, output, mix_material_info, shader_info_by_id, base_path, x_offset=0, y_offset=0, use_specific_shaders=False, addon_prefs=None):
    """
    处理混合材质，创建两个子材质并使用Mix Shader混合
    """
    # 获取混合参数
    amount_shader = mix_material_info.get('amount_shader')
    amount_value = mix_material_info.get('amount_value', 0.5)
    
    # 获取两个子材质
    material1_info = mix_material_info.get('material1')
    material2_info = mix_material_info.get('material2')
    
    if not material1_info or not material2_info:
        print("Warning: Mix Material missing material1 or material2")
        return None
    
    # 创建Mix Shader节点
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (x_offset, y_offset)
    
    # 处理Amount参数
    if amount_shader:
        # 如果Amount是着色器，递归处理
        amount_output = process_shader(nodes, links, None, amount_shader, shader_info_by_id, y_offset + 200, None, base_path)
        if amount_output:
            links.new(amount_output, mix_shader.inputs['Fac'])
    else:
        # 如果Amount是数值，直接设置
        mix_shader.inputs['Fac'].default_value = amount_value
    
    # 递归处理Material1
    material1_shader = process_material_recursive(nodes, links, material1_info, shader_info_by_id, base_path, x_offset - 300, y_offset + 300, use_specific_shaders, addon_prefs)
    if material1_shader:
        links.new(material1_shader, mix_shader.inputs[1])  # 第一个Shader输入
    
    # 递归处理Material2
    material2_shader = process_material_recursive(nodes, links, material2_info, shader_info_by_id, base_path, x_offset - 300, y_offset - 300, use_specific_shaders, addon_prefs)
    if material2_shader:
        links.new(material2_shader, mix_shader.inputs[2])  # 第二个Shader输入
    
    return mix_shader.outputs['Shader']

def process_material_recursive(nodes, links, material_info, shader_info_by_id, base_path, x_offset=0, y_offset=0, use_specific_shaders=False, addon_prefs=None):
    """
    递归处理材质，支持普通材质和混合材质
    """
    if not material_info:
        return None
    
    material_type = material_info.get('type', {}).get('id')
    
    if material_type == 1029622:  # Mix Material
        return process_mix_material(nodes, links, None, material_info, shader_info_by_id, base_path, x_offset, y_offset, use_specific_shaders, addon_prefs)
    else:
        # 普通Octane材质
        if use_specific_shaders:
            # 使用材质特定着色器
            shader_node, channel_mapping = create_material_specific_shader(nodes, links, material_type, x_offset, y_offset, addon_prefs)
            apply_material_channels_to_shader(shader_node, nodes, links, material_info, shader_info_by_id, base_path, x_offset, y_offset, channel_mapping)
        else:
            # 使用默认Principled BSDF
            shader_node = nodes.new(type='ShaderNodeBsdfPrincipled')
            shader_node.location = (x_offset, y_offset)
            apply_material_channels_to_principled(shader_node, nodes, links, material_info, shader_info_by_id, base_path, x_offset, y_offset)
        
        # 返回BSDF输出
        if 'BSDF' in shader_node.outputs:
            return shader_node.outputs['BSDF']
        else:
            return shader_node.outputs[0]  # 返回第一个输出

def create_material_specific_shader(nodes, links, material_type_id, x_offset=0, y_offset=0, addon_prefs=None):
    """
    根据Octane材质类型创建对应的Blender着色器节点
    """
    # 如果没有传递偏好设置，使用默认的Principled BSDF
    if not addon_prefs:
        shader_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        shader_node.location = (x_offset, y_offset)
        return shader_node, {
            'Diffuse': 'Base Color',
            'Roughness': 'Roughness',
            'Opacity': 'Alpha',
            'Transmission': 'Transmission Weight',
            'Normal': 'Normal',
            'Emission': 'Emission Color',
            'Matallic': 'Metallic'
        }
    
    if material_type_id == 2510 and addon_prefs.use_diffuse_shader:  # Diffuse
        shader_node = nodes.new(type='ShaderNodeBsdfDiffuse')
        shader_node.location = (x_offset, y_offset)
        return shader_node, {
            'Diffuse': 'Color',
            'Roughness': 'Roughness',
            'Normal': 'Normal'
        }
    
    elif material_type_id == 2511 and addon_prefs.use_glossy_shader:  # Glossy
        shader_node = nodes.new(type='ShaderNodeBsdfGlossy')
        shader_node.location = (x_offset, y_offset)
        return shader_node, {
            'Diffuse': 'Color',
            'Roughness': 'Roughness',  
            'Normal': 'Normal'
        }
    
    elif material_type_id == 2513 and addon_prefs.use_specular_shader:  # Specular
        shader_node = nodes.new(type='ShaderNodeBsdfGlass')
        shader_node.location = (x_offset, y_offset)
        return shader_node, {
            'Transmission': 'Color',
            'Roughness': 'Roughness',
            'Normal': 'Normal'
        }
    
    elif material_type_id == 2514 and addon_prefs.use_metallic_shader:  # Metallic
        shader_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        shader_node.location = (x_offset, y_offset)
        shader_node.inputs['Metallic'].default_value = 1.0  # 设置为金属
        return shader_node, {
            'Diffuse': 'Base Color',
            'Roughness': 'Roughness',
            'Normal': 'Normal',
            'Opacity': 'Alpha'
        }
    
    elif material_type_id == 2516 and addon_prefs.use_universal_shader:  # Universal
        shader_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        shader_node.location = (x_offset, y_offset)
        return shader_node, {
            'Diffuse': 'Base Color',
            'Roughness': 'Roughness',
            'Opacity': 'Alpha',
            'Transmission': 'Transmission Weight',
            'Normal': 'Normal',
            'Emission': 'Emission Color',
            'Matallic': 'Metallic'
        }
    
    else:  # 默认使用Principled BSDF（如果特定着色器被禁用或未识别的材质类型）
        shader_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        shader_node.location = (x_offset, y_offset)
        return shader_node, {
            'Diffuse': 'Base Color',
            'Roughness': 'Roughness',
            'Opacity': 'Alpha',
            'Transmission': 'Transmission Weight',
            'Normal': 'Normal',
            'Emission': 'Emission Color',
            'Matallic': 'Metallic'
        }

def apply_material_channels_to_shader(shader_node, nodes, links, properties, shader_info_by_id, base_path, x_offset=0, y_offset=0, channel_mapping=None):
    """
    将材质通道应用到指定的着色器节点（通用版本）
    """
    if channel_mapping is None:
        # 默认Principled BSDF映射
        channel_mapping = {
            'Diffuse': 'Base Color',
            'Roughness': 'Roughness',
            'Opacity': 'Alpha',
            'Transmission': 'Transmission Weight',
            'Normal': 'Normal',
            'Emission': 'Emission Color',
            'Matallic': 'Metallic'
        }
    
    material_type = properties.get('type', {}).get('id')
    
    # 特殊材质类型的处理
    if material_type == 2513:  # Specular
        # 设置传输权重为1.0（如果是Glass BSDF）
        if hasattr(shader_node, 'inputs') and 'IOR' in shader_node.inputs:
            shader_node.inputs['IOR'].default_value = 1.5
        
        # 从Transmission通道获取颜色值并设置为Color
        transmission_channel = properties.get('channels', {}).get('Transmission', {})
        if transmission_channel.get('used'):
            transmission_color = transmission_channel.get('color', {})
            if transmission_color and 'Color' in shader_node.inputs:
                base_color = (
                    transmission_color.get('x', 1.0),
                    transmission_color.get('y', 1.0),
                    transmission_color.get('z', 1.0),
                    1.0
                )
                shader_node.inputs['Color'].default_value = base_color
    elif material_type == 2510:  # Diffuse
        # 设置IOR值为1.0（如果适用）
        if hasattr(shader_node, 'inputs') and 'IOR' in shader_node.inputs:
            shader_node.inputs['IOR'].default_value = 1.0

    current_y_offset = y_offset

    # 处理发光
    use_emission = properties.get('use_emission', 0)
    if use_emission and 'Emission' in channel_mapping:
        emission_info = properties.get('emission', {})
        if emission_info:
            emission_type = emission_info.get('emission_type')
            if emission_type == 'Blackbody':
                shader_output = process_blackbody_emission(
                    nodes, links, shader_node,
                    emission_info,
                    shader_info_by_id,
                    current_y_offset,
                    'Emission Color',
                    base_path
                )
                if shader_output and 'Emission Color' in shader_node.inputs:
                    links.new(shader_output, shader_node.inputs['Emission Color'])
                    if 'Emission Strength' in shader_node.inputs:
                        shader_node.inputs['Emission Strength'].default_value = 1.0
            elif emission_type == 'Texture':
                shader_output = process_texture_emission(
                    nodes, links, shader_node,
                    emission_info,
                    shader_info_by_id,
                    current_y_offset,
                    'Emission Color',
                    base_path
                )
                if shader_output and 'Emission Color' in shader_node.inputs:
                    links.new(shader_output, shader_node.inputs['Emission Color'])
                    emission_strength = emission_info.get('emission_strength', 1.0)
                    if 'Emission Strength' in shader_node.inputs:
                        shader_node.inputs['Emission Strength'].default_value = emission_strength

    # 处理材质通道
    for prop, input_name in channel_mapping.items():
        channel = properties.get('channels', {}).get(prop, {})
        if not channel.get('used', False):
            continue

        link = channel.get('link', {})
        
        # Handle gradient image path if provided
        gradient_image_path = link.get('gradient', {}).get('image_path') if link else None
        if gradient_image_path and os.path.exists(gradient_image_path):
            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.location = (x_offset - 300, current_y_offset)
            img = bpy.data.images.load(gradient_image_path)
            tex_node.image = img
            
            if input_name in ['Color', 'Base Color', 'Emission']:
                tex_node.image.colorspace_settings.name = 'sRGB'
            else:
                tex_node.image.colorspace_settings.name = 'Non-Color'
            
            # 创建映射节点
            mapping_node = nodes.new(type='ShaderNodeMapping')
            mapping_node.location = (x_offset - 600, current_y_offset)
                
            tex_coord_node = nodes.get('Texture Coordinate')
            if not tex_coord_node:
                tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
                tex_coord_node.location = (x_offset - 900, current_y_offset)
            
            # 连接: Texture Coordinate → Mapping → Image Texture
            links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
            links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
            
            shader_output = tex_node.outputs['Color']
        else:
            # 处理其他类型的着色器
            shader_output = process_shader(nodes, links, shader_node, link, shader_info_by_id, current_y_offset, input_name, base_path)

        if not shader_output:
            # 处理颜色和浮点值
            color_data = channel.get('color')
            float_value = channel.get('float_value')
            
            if prop == 'Matallic' and float_value is not None and 'Metallic' in shader_node.inputs:
                shader_node.inputs['Metallic'].default_value = float_value
                continue
                
            if color_data:
                shader_output = process_color_shader(
                    nodes, links, shader_node, {'color': color_data}, shader_info_by_id, current_y_offset, input_name, base_path
                )
            elif 'float_texture_value' in channel or 'float_value' in channel:
                shader_output = process_float_texture(
                    nodes, links, shader_node, channel, shader_info_by_id, current_y_offset, input_name, base_path
                )

        if shader_output and input_name in shader_node.inputs:
            # 处理特殊连接
            if input_name.lower() == 'normal':
                normal_map = nodes.new(type='ShaderNodeNormalMap')
                normal_map.location = (x_offset - 150, current_y_offset)
                links.new(shader_output, normal_map.inputs['Color'])
                links.new(normal_map.outputs['Normal'], shader_node.inputs[input_name])
            else:
                links.new(shader_output, shader_node.inputs[input_name])

        current_y_offset -= 300

def apply_material_channels_to_principled(principled, nodes, links, properties, shader_info_by_id, base_path, x_offset=0, y_offset=0):
    """
    将材质通道应用到指定的Principled BSDF节点
    """
    material_type = properties.get('type', {}).get('id')
    
    if material_type == 2513:
        # 设置传输权重为1.0
        principled.inputs['Transmission Weight'].default_value = 1.0
        
        # 从Transmission通道获取颜色值并设置为Base Color
        transmission_channel = properties.get('channels', {}).get('Transmission', {})
        if transmission_channel.get('used'):
            transmission_color = transmission_channel.get('color', {})
            if transmission_color:
                base_color = (
                    transmission_color.get('x', 1.0),
                    transmission_color.get('y', 1.0),
                    transmission_color.get('z', 1.0),
                    1.0
                )
                # 设置Base Color
                principled.inputs['Base Color'].default_value = base_color
                
                # 更新Diffuse通道的值以防止后续处理覆盖
                if 'channels' in properties and 'Diffuse' in properties['channels']:
                    properties['channels']['Diffuse']['color'] = transmission_color
    elif material_type == 2510:  # Diffuse材质
        # 设置IOR值为1.0
        principled.inputs['IOR'].default_value = 1.0

    channel_mapping = {
        'Diffuse': 'Base Color',
        'Roughness': 'Roughness',
        'Opacity': 'Alpha',
        'Transmission': 'Transmission Weight',
        'Bump': 'Normal',
        'Normal': 'Normal',
        'Emission': 'Emission Color',
        'Displacement': 'Displacement',
        'Matallic': 'Metallic',  # 修改这里以匹配JSON中的拼写
    }

    current_y_offset = y_offset

    # Check if emission is enabled
    use_emission = properties.get('use_emission', 0)
    if use_emission:
        emission_info = properties.get('emission', {})
        if emission_info:
            emission_type = emission_info.get('emission_type')
            if emission_type == 'Blackbody':
                # Process blackbody emission
                shader_output = process_blackbody_emission(
                    nodes, links, principled,
                    emission_info,
                    shader_info_by_id,
                    current_y_offset,
                    'Emission Color',
                    base_path
                )
                if shader_output:
                    links.new(shader_output, principled.inputs['Emission Color'])
                    # Set emission strength
                    principled.inputs['Emission Strength'].default_value = 1.0
            elif emission_type == 'Texture':
                # Process texture emission
                shader_output = process_texture_emission(
                    nodes, links, principled,
                    emission_info,
                    shader_info_by_id,
                    current_y_offset,
                    'Emission Color',
                    base_path
                )
                if shader_output:
                    links.new(shader_output, principled.inputs['Emission Color'])
                    emission_strength = emission_info.get('emission_strength', 1.0)
                    principled.inputs['Emission Strength'].default_value = emission_strength

    for prop, input_name in channel_mapping.items():
        # Get the current channel data
        channel = properties.get('channels', {}).get(prop, {})
        if not channel.get('used', False):
            continue  # Skip unused channels

        link = channel.get('link', {})
        
        # Handle gradient image path if provided
        gradient_image_path = link.get('gradient', {}).get('image_path') if link else None
        if gradient_image_path and os.path.exists(gradient_image_path):
            # Create a texture node using the gradient image
            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.location = (x_offset - 300, current_y_offset)
            img = bpy.data.images.load(gradient_image_path)
            tex_node.image = img
            # Set color space based on input_name
            if input_name in ['Base Color', 'Emission']:
                tex_node.image.colorspace_settings.name = 'sRGB'
            else:
                tex_node.image.colorspace_settings.name = 'Non-Color'
            
            # 创建映射节点
            mapping_node = nodes.new(type='ShaderNodeMapping')
            mapping_node.location = (x_offset - 600, current_y_offset)
            
            # Add a Texture Coordinate node if needed
            tex_coord_node = nodes.get('Texture Coordinate')
            if not tex_coord_node:
                tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
                tex_coord_node.location = (x_offset - 900, current_y_offset)
            
            # 连接: Texture Coordinate → Mapping → Image Texture
            links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
            links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
            
            shader_output = tex_node.outputs['Color']
        else:
            # 处理其他类型的着色器
            shader_output = process_shader(nodes, links, principled, link, shader_info_by_id, current_y_offset, input_name, base_path)

        if not shader_output:
            # Handle cases where shader_output is None
            color_data = channel.get('color')
            float_value = channel.get('float_value')
            use_color = True
            
            if color_data:
                color_vec = vector_to_dict(color_data)
                if prop == 'Diffuse':
                    if color_vec['x'] == 0.9 and color_vec['y'] == 0.9 and color_vec['z'] == 0.9:
                        use_color = True
                else:
                    if color_vec['x'] == 0.0 and color_vec['y'] == 0.0 and color_vec['z'] == 0.0:
                        use_color = False
            else:
                use_color = False

            # Handle float values directly for Metallic
            if prop == 'Metallic' and float_value is not None:
                principled.inputs['Metallic'].default_value = float_value
                continue

            if use_color:
                shader_output = process_color_shader(
                    nodes, links, principled, {'color': color_data}, shader_info_by_id, current_y_offset, input_name, base_path
                )
            elif 'float_texture_value' in channel or 'float_value' in channel:
                shader_output = process_float_texture(
                    nodes, links, principled, channel, shader_info_by_id, current_y_offset, input_name, base_path
                )

        if shader_output:
            # 根据输入名称决定如何连接 shader 输出
            if input_name.lower() == 'normal':
                normal_map = nodes.new(type='ShaderNodeNormalMap')
                normal_map.location = (-150, current_y_offset)
                links.new(shader_output, normal_map.inputs['Color'])
                links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
            elif input_name.lower() == 'bump':
                bump_node = nodes.new(type='ShaderNodeBump')
                bump_node.location = (-150, current_y_offset)
                links.new(shader_output, bump_node.inputs['Height'])
                links.new(bump_node.outputs['Normal'], principled.inputs['Normal'])
            elif input_name.lower() == 'emission color':
                links.new(shader_output, principled.inputs['Emission Color'])
                emission_strength = channel.get('emission_strength', 1.0)
                principled.inputs['Emission Strength'].default_value = emission_strength
            elif input_name in principled.inputs:
                links.new(shader_output, principled.inputs[input_name])
            else:
                print(f"Warning: Input '{input_name}' not found in Principled BSDF")

        else:
            print(f"Warning: Could not process channel '{prop}' for material")

        current_y_offset -= 300

def apply_material_properties(materials, shader_info_by_id, base_path):
    """Applies material properties to Blender's material system."""
    # 获取用户偏好设置
    addon_prefs = bpy.context.preferences.addons[__name__].preferences
    use_specific_shaders = addon_prefs.use_material_specific_shaders
    
    for mat_name, properties in materials.items():
        # 检查是否有"未找到材质标签"的消息
        if properties.get('message') == "对象上未找到材质标签。":
            print(f"Skipping object '{properties.get('object', 'Unknown')}' (no material tags)")
            continue

        mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear existing nodes
        nodes.clear()

        # Create Output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)

        material_type = properties.get('type', {}).get('id')
        
        # 检查是否是混合材质
        if material_type == 1029622:  # Mix Material
            # 处理混合材质
            mix_shader_output = process_mix_material(nodes, links, output, properties, shader_info_by_id, base_path, 0, 0, use_specific_shaders, addon_prefs)
            if mix_shader_output:
                links.new(mix_shader_output, output.inputs['Surface'])
        else:
            # 普通材质处理
            if use_specific_shaders:
                # 使用材质特定着色器
                shader_node, channel_mapping = create_material_specific_shader(nodes, links, material_type, 0, 0, addon_prefs)
                apply_material_channels_to_shader(shader_node, nodes, links, properties, shader_info_by_id, base_path, 0, 0, channel_mapping)
                # 连接到材质输出
                if 'BSDF' in shader_node.outputs:
                    links.new(shader_node.outputs['BSDF'], output.inputs['Surface'])
                else:
                    links.new(shader_node.outputs[0], output.inputs['Surface'])
            else:
                # 使用默认Principled BSDF
                principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                principled.location = (0, 0)
                links.new(principled.outputs['BSDF'], output.inputs['Surface'])
                
                # 应用材质通道
                apply_material_channels_to_principled(principled, nodes, links, properties, shader_info_by_id, base_path)

        # 打印材质类型信息
        material_type_info = properties.get('type', {})
        type_name = material_type_info.get('name', 'Unknown')
        print(f"Material '{mat_name}' ({type_name}) processed")

        # Assign material to object if object_name is provided
        obj_name = properties.get('object_name')
        if obj_name:
            # Ensure the object name matches Blender's encoding (handle non-English names)
            obj = bpy.data.objects.get(obj_name)
            if obj and hasattr(obj, 'data') and hasattr(obj.data, 'materials'):
                # Assign the material to the object
                if mat_name not in obj.data.materials:
                    obj.data.materials.append(mat)
            else:
                print(f"Object '{obj_name}' not found or is not a mesh object.")



def main():
    # 使用系统缓存目录
    cache_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'cache')
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 修改 JSON 文件路径到缓存目录
    input_file_path = os.path.join(cache_dir, 'octane_material_info.json')
    
    if not os.path.exists(input_file_path):
        return
    
    # 解析 JSON 文件
    materials, shader_info_by_id = parse_material_info(input_file_path)
    
    if not materials:
        return
    
    # 使用缓存目录作为基础路径
    base_path = cache_dir
    
    # 应用材质属性
    apply_material_properties(materials, shader_info_by_id, base_path)
    
class Cinema4DPreferences(AddonPreferences):
    bl_idname = __name__
    
    # 总体开关
    use_material_specific_shaders: bpy.props.BoolProperty(
        name="使用材质特定着色器",
        description="根据Octane材质类型创建对应的Blender着色器节点",
        default=False
    )
    
    # 各材质类型的单独开关
    use_diffuse_shader: bpy.props.BoolProperty(
        name="Diffuse → Diffuse BSDF",
        description="使用Diffuse BSDF替代Principled BSDF处理Diffuse材质",
        default=True
    )
    
    use_glossy_shader: bpy.props.BoolProperty(
        name="Glossy → Glossy BSDF", 
        description="使用Glossy BSDF替代Principled BSDF处理Glossy材质",
        default=True
    )
    
    use_specular_shader: bpy.props.BoolProperty(
        name="Specular → Glass BSDF",
        description="使用Glass BSDF替代Principled BSDF处理Specular材质", 
        default=True
    )
    
    use_metallic_shader: bpy.props.BoolProperty(
        name="Metallic → Principled (金属)",
        description="使用预设金属的Principled BSDF处理Metallic材质",
        default=True
    )
    
    use_universal_shader: bpy.props.BoolProperty(
        name="Universal → Principled BSDF",
        description="使用Principled BSDF处理Universal材质",
        default=True
    )

    def draw(self, context):
        layout = self.layout
        
        # 总体开关
        layout.prop(self, "use_material_specific_shaders")
        
        # 如果启用了材质特定着色器，显示详细控制
        if self.use_material_specific_shaders:
            box = layout.box()
            box.label(text="材质类型控制:", icon='MATERIAL')
            
            # 创建两列布局
            col = box.column(align=True)
            col.prop(self, "use_diffuse_shader")
            col.prop(self, "use_glossy_shader") 
            col.prop(self, "use_specular_shader")
            col.prop(self, "use_metallic_shader")
            col.prop(self, "use_universal_shader")
            
            box.separator()
            box.label(text="注意: 未勾选的材质类型将使用默认的Principled BSDF", icon='INFO')
        
        layout.separator()

        # 第一行按钮
        row = layout.row()
        row.operator(
            "wm.url_open", 
            text="查看文档",
            icon='HELP'
        ).url = "https://www.yuque.com/shouwangxingkong-0p4w3/ldvruc/wn8pstg9pwd6r6mp?singleDoc"
        row.operator(
            "wm.url_open", 
            text="关于作者",
            icon='USER'
        ).url = "https://space.bilibili.com/34368968"

        # 第二行按钮
        row = layout.row()
        row.operator(
            "wm.url_open", 
            text="检查更新",
            icon='FILE_REFRESH'
        ).url = "https://www.yuque.com/shouwangxingkong-0p4w3/ldvruc/gmd4pud4fu2vz30z?singleDoc"
        row.operator(
            "wm.url_open", 
            text="加入QQ群",
            icon='COMMUNITY'
        ).url = "https://qm.qq.com/cgi-bin/qm/qr?k=9KgmVUQMfoGf7g_s-4tSe15oMJ6rbz6b&jump_from=webapi&authKey=hs9XWuCbT1jx9ytpzSsXbJuQCwUc2kXy0gRJfA+qMaVoXTbvhiOKz0dHOnP1+Cvt"

        # 第三行按钮（单个）
        row = layout.row()
        row.operator(
            "wm.url_open", 
            text="购买付费版",
            icon='FUND'
        ).url = "https://www.bilibili.com/video/BV1x2pSeNEaV/"



class IMPORT_OT_OctaneMaterial(bpy.types.Operator):
    bl_idname = "import_octane_material.importx"
    bl_label = "Import Octane Material"
    bl_description = "Import materials from Octane"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            main()
            self.report({'INFO'}, "Materials imported successfully")
        except Exception as e:
            self.report({'ERROR'}, f"Error importing materials: {str(e)}")
        return {'FINISHED'}


class IMPORT_PT_OctaneMaterialPanel(bpy.types.Panel):
    bl_label = "Import Octane Material"
    bl_idname = "IMPORT_PT_OctaneMaterialPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Material"

    def draw(self, context):
        layout = self.layout
        
        # 获取插件偏好设置
        addon_prefs = context.preferences.addons[__name__].preferences
        
        # 添加材质特定着色器选项
        layout.prop(addon_prefs, "use_material_specific_shaders")
        
        # 如果启用了材质特定着色器，显示详细控制
        if addon_prefs.use_material_specific_shaders:
            box = layout.box()
            box.label(text="材质类型控制:", icon='MATERIAL')
            
            # 创建紧凑的布局
            col = box.column(align=True)
            
            # 显示每种材质类型的状态
            row = col.row()
            row.prop(addon_prefs, "use_diffuse_shader", text="Diffuse")
            row = col.row()
            row.prop(addon_prefs, "use_glossy_shader", text="Glossy")
            row = col.row()
            row.prop(addon_prefs, "use_specular_shader", text="Specular")
            row = col.row()
            row.prop(addon_prefs, "use_metallic_shader", text="Metallic")
            row = col.row()
            row.prop(addon_prefs, "use_universal_shader", text="Universal")
            
            # 显示当前激活的着色器类型
            active_shaders = []
            if addon_prefs.use_diffuse_shader:
                active_shaders.append("Diffuse→DiffuseBSDF")
            if addon_prefs.use_glossy_shader:
                active_shaders.append("Glossy→GlossyBSDF")
            if addon_prefs.use_specular_shader:
                active_shaders.append("Specular→GlassBSDF")
            if addon_prefs.use_metallic_shader:
                active_shaders.append("Metallic→Principled(金属)")
            if addon_prefs.use_universal_shader:
                active_shaders.append("Universal→Principled")
            
            if not active_shaders:
                box.label(text="所有材质将使用 Principled BSDF", icon='INFO')
            else:
                sub_box = box.box()
                sub_box.label(text="激活的特定着色器:", icon='CHECKMARK')
                for shader in active_shaders:
                    sub_box.label(text="• " + shader)
                
                disabled_types = []
                if not addon_prefs.use_diffuse_shader:
                    disabled_types.append("Diffuse")
                if not addon_prefs.use_glossy_shader:
                    disabled_types.append("Glossy")
                if not addon_prefs.use_specular_shader:
                    disabled_types.append("Specular")
                if not addon_prefs.use_metallic_shader:
                    disabled_types.append("Metallic")
                if not addon_prefs.use_universal_shader:
                    disabled_types.append("Universal")
                
                if disabled_types:
                    sub_box.label(text="使用默认Principled BSDF:", icon='X')
                    for disabled in disabled_types:
                        sub_box.label(text="• " + disabled)
        else:
            layout.label(text="使用统一的 Principled BSDF", icon='NODE')
        
        layout.separator()
        layout.operator("import_octane_material.importx")
        


def register():
    bpy.utils.register_class(Cinema4DPreferences)
    bpy.utils.register_class(IMPORT_OT_OctaneMaterial)
    bpy.utils.register_class(IMPORT_PT_OctaneMaterialPanel)


def unregister():
    bpy.utils.unregister_class(Cinema4DPreferences)
    bpy.utils.unregister_class(IMPORT_OT_OctaneMaterial)
    bpy.utils.unregister_class(IMPORT_PT_OctaneMaterialPanel)


if __name__ == "__main__":
    register()