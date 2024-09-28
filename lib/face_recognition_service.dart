import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

typedef DetectFacesC = Int32 Function(
    Pointer<Utf8> imagePath,
    Pointer<Utf8> yolov5ModelPath,
    Pointer<Utf8> arcfaceModelPath,
    Pointer<Int32> faces,
    Int32 maxFaces,
    Pointer<Pointer<Uint8>> faceData,
    Pointer<Int32> faceDataSizes,
    Pointer<Pointer<Float>> faceFeatures);

typedef DetectFacesDart = int Function(
    Pointer<Utf8> imagePath,
    Pointer<Utf8> yolov5ModelPath,
    Pointer<Utf8> arcfaceModelPath,
    Pointer<Int32> faces,
    int maxFaces,
    Pointer<Pointer<Uint8>> faceData,
    Pointer<Int32> faceDataSizes,
    Pointer<Pointer<Float>> faceFeatures);

typedef CompareFacesC = Float Function(
    Pointer<Float> features1, Pointer<Float> features2, Int32 featureSize);

typedef CompareFacesDart = double Function(
    Pointer<Float> features1, Pointer<Float> features2, int featureSize);

class FaceRecognitionService {
  static late DynamicLibrary _lib;
  static late DetectFacesDart _detectFaces;
  static late CompareFacesDart _compareFaces;
  static bool _isInitialized = false;
  static late String _dllPath;
  static late String _yolov5ModelPath;
  static late String _arcfaceModelPath;

  static bool get isInitialized => _isInitialized;

  static Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      print("正在初始化 FaceRecognitionService");
      await _copyAssetsToLocal();
      _dllPath = await _getLocalPath('face_recognition.dll');
      _yolov5ModelPath = await _getLocalPath('yolov5l.onnx');
      _arcfaceModelPath = await _getLocalPath('arcface_model.onnx');

      print("DLL 路径: $_dllPath");
      print("YOLOV5 模型路径: $_yolov5ModelPath");
      print("ArcFace 模型路径: $_arcfaceModelPath");

      print("正在检查DLL文件是否存在...");
      final dllFile = File(_dllPath);
      if (!await dllFile.exists()) {
        throw Exception("DLL文件不存在: $_dllPath");
      }

      print("正在尝试加载DLL...");
      try {
        _lib = DynamicLibrary.open(_dllPath);
        print("成功加载 DLL: $_dllPath");
      } catch (e) {
        print("加载 DLL 失败: $e");
        rethrow;
      }

      _detectFaces = _lib
          .lookup<NativeFunction<DetectFacesC>>('detect_faces')
          .asFunction();
      _compareFaces = _lib
          .lookup<NativeFunction<CompareFacesC>>('compare_faces')
          .asFunction();

      _isInitialized = true;
      print("FaceRecognitionService 初始化成功");
    } catch (e) {
      print('FaceRecognitionService 初始化失败: $e');
      rethrow;
    }
  }

  static Future<List<FaceData>> recognizeFaces(List<String> imagePaths) async {
    final List<FaceData> allFaces = [];
    for (final imagePath in imagePaths) {
      try {
        print("正在处理图像: $imagePath");
        final faces = await _detectFacesInImage(imagePath);
        allFaces.addAll(faces);
        print("在 $imagePath 中检测到 ${faces.length} 个人脸");
      } catch (e) {
        print("处理图像 $imagePath 时出错: $e");
      }
    }
    return allFaces;
  }

  static Future<List<FaceData>> _detectFacesInImage(String imagePath) async {
    final normalizedPath = _normalizePath(imagePath);
    print("正在处理图像路径: $normalizedPath");

    // 检查图像文件是否存在
    if (!await File(normalizedPath).exists()) {
      print("图像文件不存在: $normalizedPath");
      return [];
    }

    // 检查模型文件是否存在
    if (!await File(_yolov5ModelPath).exists()) {
      print("YOLOV5 模型文件不存在: $_yolov5ModelPath");
      return [];
    }
    if (!await File(_arcfaceModelPath).exists()) {
      print("ArcFace 模型文件不存在: $_arcfaceModelPath");
      return [];
    }

    final maxFaces = 10;
    final facesPtr = calloc<Int32>(maxFaces * 4);
    final faceDataPtr = calloc<Pointer<Uint8>>(maxFaces);
    final faceDataSizesPtr = calloc<Int32>(maxFaces);
    final faceFeaturesPtr = calloc<Pointer<Float>>(maxFaces);

    try {
      print("YOLOV5 模型路径: $_yolov5ModelPath");
      print("ArcFace 模型路径: $_arcfaceModelPath");

      final result = _detectFaces(
        normalizedPath.toNativeUtf8(),
        _yolov5ModelPath.toNativeUtf8(),
        _arcfaceModelPath.toNativeUtf8(),
        facesPtr,
        maxFaces,
        faceDataPtr,
        faceDataSizesPtr,
        faceFeaturesPtr,
      );

      print("检测到 $result 个人脸");

      if (result < 0) {
        print("人脸检测失败，错误代码: $result");
        return [];
      }

      final List<FaceData> detectedFaces = [];

      for (int i = 0; i < result; i++) {
        final faceImage = faceDataPtr[i].asTypedList(faceDataSizesPtr[i]);
        final features = faceFeaturesPtr[i].asTypedList(128);

        detectedFaces.add(FaceData(
          faceImage: Uint8List.fromList(faceImage),
          features: Float32List.fromList(features),
        ));

        calloc.free(faceDataPtr[i]);
        calloc.free(faceFeaturesPtr[i]);
      }

      return detectedFaces;
    } catch (e) {
      print("_detectFacesInImage 出错: $e");
      return [];
    } finally {
      calloc.free(facesPtr);
      calloc.free(faceDataPtr);
      calloc.free(faceDataSizesPtr);
      calloc.free(faceFeaturesPtr);
    }
  }

  static double compareFaces(Float32List features1, Float32List features2) {
    final ptr1 = calloc<Float>(features1.length);
    final ptr2 = calloc<Float>(features2.length);

    for (int i = 0; i < features1.length; i++) {
      ptr1[i] = features1[i];
      ptr2[i] = features2[i];
    }

    final similarity = _compareFaces(ptr1, ptr2, features1.length);

    calloc.free(ptr1);
    calloc.free(ptr2);

    return similarity;
  }

  static Future<List<FaceData>> detectFacesInImage(String imagePath) async {
    final imagePathNative = imagePath.toNativeUtf8();
    final yolov5ModelPathNative = _yolov5ModelPath.toNativeUtf8();
    final arcfaceModelPathNative = _arcfaceModelPath.toNativeUtf8();

    final faces = calloc<Int32>(4 * 10);
    final faceData = calloc<Pointer<Uint8>>(10);
    final faceDataSizes = calloc<Int32>(10);
    final faceFeatures = calloc<Pointer<Float>>(10);

    try {
      final numFaces = _detectFaces(
        imagePathNative,
        yolov5ModelPathNative,
        arcfaceModelPathNative,
        faces,
        10,
        faceData,
        faceDataSizes,
        faceFeatures,
      );

      if (numFaces < 0) {
        throw Exception('Face detection failed');
      }

      List<FaceData> detectedFaces = [];
      for (int i = 0; i < numFaces; i++) {
        final faceImageData = faceData[i].asTypedList(faceDataSizes[i]);
        final faceFeatureData = faceFeatures[i].asTypedList(128);

        detectedFaces.add(FaceData(
          faceImage: Uint8List.fromList(faceImageData),
          features: Float32List.fromList(faceFeatureData),
        ));
      }

      return detectedFaces;
    } finally {
      calloc.free(imagePathNative);
      calloc.free(yolov5ModelPathNative);
      calloc.free(arcfaceModelPathNative);
      calloc.free(faces);
      for (var i = 0; i < 10; i++) {
        if (faceData[i] != nullptr) calloc.free(faceData[i]);
        if (faceFeatures[i] != nullptr) calloc.free(faceFeatures[i]);
      }
      calloc.free(faceData);
      calloc.free(faceDataSizes);
      calloc.free(faceFeatures);
    }
  }

  static Future<void> cleanupTempFiles() async {
    final directory = await getApplicationDocumentsDirectory();
    final files = directory.listSync();
    for (var file in files) {
      if (file is File &&
          (file.path.endsWith('.onnx') || file.path.endsWith('.dll'))) {
        await file.delete();
      }
    }
  }

  static String _normalizePath(String path) {
    return path.replaceAll('\\', '/');
  }

  static Future<void> _copyAssetsToLocal() async {
    final directory = await getApplicationDocumentsDirectory();
    final assets = [
      'face_recognition.dll',
      'yolov5l.onnx',
      'arcface_model.onnx'
    ];

    for (final asset in assets) {
      final file = File('${directory.path}${Platform.pathSeparator}$asset');
      print("正在复制资源文件: $asset");
      final byteData = await rootBundle.load('assets/$asset');
      await file.writeAsBytes(byteData.buffer.asUint8List());
      print("资源文件复制完成: $asset");
    }
  }

  static Future<String> _getLocalPath(String fileName) async {
    final directory = await getApplicationDocumentsDirectory();
    return '${directory.path}${Platform.pathSeparator}$fileName'; // 使用平台特定的路径分隔符
  }

  static String _encodePath(String path) {
    return Uri.encodeFull(path);
  }

  static Future<String> readLogFile() async {
    final directory = await getApplicationDocumentsDirectory();
    final logFile = File(
        '${directory.path}${Platform.pathSeparator}face_recognition_log.txt');
    if (await logFile.exists()) {
      return await logFile.readAsString();
    }
    return 'Log file not found';
  }

  static Future<void> clearLogFile() async {
    final directory = await getApplicationDocumentsDirectory();
    final logFile = File(
        '${directory.path}${Platform.pathSeparator}face_recognition_log.txt');
    if (await logFile.exists()) {
      await logFile.delete();
    }
  }
}

class FaceData {
  final Uint8List faceImage;
  final Float32List features;

  FaceData({required this.faceImage, required this.features});
}
