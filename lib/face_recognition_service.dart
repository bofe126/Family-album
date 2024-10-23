import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:async';
import 'logger.dart'; // 导入您的 logger.dart 文件

typedef DetectFacesNative = Int32 Function(
    Pointer<Utf8> imagePath,
    Pointer<Int32> faces,
    Pointer<Pointer<Uint8>> faceData,
    Pointer<Int32> faceDataSizes,
    Pointer<Pointer<Float>> faceFeatures,
    Int32 maxFaces,
    Float scoreThreshold
);

typedef DetectFaces = int Function(
    Pointer<Utf8> imagePath,
    Pointer<Int32> faces,
    Pointer<Pointer<Uint8>> faceData,
    Pointer<Int32> faceDataSizes,
    Pointer<Pointer<Float>> faceFeatures,
    int maxFaces,
    double scoreThreshold
);

typedef CompareFacesNative = Float Function(
    Pointer<Float> features1,
    Pointer<Float> features2,
    Int32 featureSize
);

typedef CompareFaces = double Function(
    Pointer<Float> features1,
    Pointer<Float> features2,
    int featureSize
);

class FaceRecognitionConfig {
  static const String dllPath = 'face_recognition.dll';
  static const String yolov5ModelPath = 'assets/yolov5s-face.onnx';
  static const String arcfaceModelPath = 'assets/arcface_model.onnx';
  static const int maxFaces = 10;
}

class FaceRecognitionException implements Exception {
  final String message;
  FaceRecognitionException(this.message);
  @override
  String toString() => 'FaceRecognitionException: $message';
}

class FaceRecognitionService {
  static final _logger = logger; // 这里使用从 logger.dart 导入的全局 logger 实例
  static late DynamicLibrary _lib;
  static late DetectFaces _detectFaces;
  static late CompareFaces _compareFaces;
  static bool _isInitialized = false;

  static bool get isInitialized => _isInitialized;

  static Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      _logger.i('FaceRecognitionService: initialization_start');

      _lib = DynamicLibrary.open(FaceRecognitionConfig.dllPath);
      _detectFaces = _lib
          .lookup<NativeFunction<DetectFacesNative>>('detect_faces')
          .asFunction();
      _compareFaces = _lib
          .lookup<NativeFunction<CompareFacesNative>>('compare_faces')
          .asFunction();

      _isInitialized = true;
      _logger.i('FaceRecognitionService: initialization_success');
    } catch (e) {
      _logger.e(
          'FaceRecognitionService: initialization_failure - ${e.toString()}');
      rethrow;
    }
  }

  static Future<List<FaceData>> recognizeFaces(List<String> imagePaths) async {
    _logger.i('Face recognition: start=${imagePaths.length}');
    final futures = imagePaths.map((path) => _detectFacesInImage(path));
    final results = await Future.wait(futures);
    final allFaces = results.expand((faces) => faces).toList();
    _logger.i('Face recognition: complete=${allFaces.length}');
    return allFaces;
  }

  static Future<List<FaceData>> _detectFacesInImage(String imagePath) async {
    final stopwatch = Stopwatch()..start();
    try {
      final normalizedPath = _normalizePath(imagePath);
      _logger.i('Face detection: start=$normalizedPath');

      final List<FaceData> detectedFaces = [];

      using((arena) {
        final facesPtr = arena<Int32>(FaceRecognitionConfig.maxFaces * 4);
        final faceDataPtr = arena<Pointer<Uint8>>(FaceRecognitionConfig.maxFaces);
        final faceDataSizesPtr = arena<Int32>(FaceRecognitionConfig.maxFaces);
        final faceFeaturesPtr = arena<Pointer<Float>>(FaceRecognitionConfig.maxFaces);

        final result = _detectFaces(
          normalizedPath.toNativeUtf8(allocator: arena),
          facesPtr,
          faceDataPtr,
          faceDataSizesPtr,
          faceFeaturesPtr,
          FaceRecognitionConfig.maxFaces,
          0.6  // 默认阈值
        );

        _logger.i('Face detection: result=$result');

        if (result < 0) {
          throw FaceRecognitionException("人脸检测失败，错误代码: $result");
        }

        for (int i = 0; i < result; i++) {
          final faceImage = faceDataPtr[i].asTypedList(faceDataSizesPtr[i]);
          final features = faceFeaturesPtr[i].asTypedList(128);

          detectedFaces.add(FaceData(
            faceImage: Uint8List.fromList(faceImage),
            features: Float32List.fromList(features),
          ));

          arena.free(faceDataPtr[i]);
          arena.free(faceFeaturesPtr[i]);
        }
      });

      _logger.i('Face detection: processed=${detectedFaces.length}');
      return detectedFaces;
    } catch (e) {
      _logger.e('Face detection: error=${e.toString()}');
      rethrow;
    } finally {
      stopwatch.stop();
      _logger.i('Face detection: time=${stopwatch.elapsedMilliseconds}ms');
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

  static Future<List<FaceData>> detectFaces(String imagePath,
      {int maxFaces = 10, double scoreThreshold = 0.6}) async {
    // 添加参数验证
    if (maxFaces <= 0 || maxFaces > 100) {
        _logger.w('Invalid maxFaces ($maxFaces), using default value 10');
        maxFaces = 10;
    }

    List<FaceData> detectedFaces = [];

    await using((Arena arena) {
      final imagePathNative = imagePath.toNativeUtf8(allocator: arena);
      final faces = arena<Int32>(4 * maxFaces);  // 确保分配足够的空间
      final faceData = arena<Pointer<Uint8>>(maxFaces);
      final faceDataSizes = arena<Int32>(maxFaces);
      final faceFeatures = arena<Pointer<Float>>(maxFaces);

      _logger.i('调用detect_faces: path=$imagePath, maxFaces=$maxFaces, threshold=$scoreThreshold');
      
      final numFaces = _detectFaces(
        imagePathNative,
        faces,
        faceData,
        faceDataSizes,
        faceFeatures,
        maxFaces,
        scoreThreshold
      );

      if (numFaces < 0) {
        _logger.e('人脸检测失败: $numFaces');
        throw Exception('Face detection failed');
      }

      // 添加边界检查
      for (int i = 0; i < numFaces && i < maxFaces; i++) {
        if (faceData[i] != nullptr && faceFeatures[i] != nullptr) {
          final faceImageData = faceData[i].asTypedList(faceDataSizes[i]);
          final faceFeatureData = faceFeatures[i].asTypedList(128);

          detectedFaces.add(FaceData(
            faceImage: Uint8List.fromList(faceImageData),
            features: Float32List.fromList(faceFeatureData),
          ));
        }
      }
    });

    return detectedFaces;
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
    _lib.close(); // 关闭动态库
    _isInitialized = false; // 重置初始化状态
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
      _logger.i("Asset copy: start=$asset");
      final byteData = await rootBundle.load('assets/$asset');
      await file.writeAsBytes(byteData.buffer.asUint8List());
      _logger.i("Asset copy: complete=$asset");
    }
  }

  static Future<String> _getLocalPath(String fileName) async {
    final directory = await getApplicationDocumentsDirectory();
    return '${directory.path}${Platform.pathSeparator}$fileName'; // 使用平台特定的路径分隔符
  }

  static String _encodePath(String path) {
    return Uri.encodeFull(path);
  }
}

class FaceData {
  final Uint8List faceImage;
  final Float32List features;

  FaceData({required this.faceImage, required this.features});
}
