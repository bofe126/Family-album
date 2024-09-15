import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

typedef DetectFacesC = Int32 Function(
    Pointer<Utf8> imagePath,
    Pointer<Utf8> prototxtPath,
    Pointer<Utf8> caffeModelPath,
    Pointer<Utf8> arcfaceModelPath,
    Pointer<Int32> faces,
    Int32 maxFaces,
    Pointer<Pointer<Uint8>> faceData,
    Pointer<Int32> faceDataSizes,
    Pointer<Pointer<Float>> faceFeatures);

typedef DetectFacesDart = int Function(
    Pointer<Utf8> imagePath,
    Pointer<Utf8> prototxtPath,
    Pointer<Utf8> caffeModelPath,
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
  static const MethodChannel _channel = MethodChannel('face_recognition');
  static late DynamicLibrary _lib;
  static late DetectFacesDart _detectFaces;
  static late CompareFacesDart _compareFaces;
  static bool _isInitialized = false;
  static late String _dllPath;
  static late String _yolov5ModelPath;
  static late String _arcfaceModelPath;

  static Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      await _copyAssetsToLocal();
      _dllPath = 'assets/face_recognition.dll';
      _yolov5ModelPath = await _getLocalPath('yolov5-face.onnx');
      _arcfaceModelPath = await _getLocalPath('arcface_model.onnx');

      _lib = DynamicLibrary.open(_dllPath);
      _detectFaces = _lib
          .lookup<NativeFunction<DetectFacesC>>('detect_faces')
          .asFunction();
      _compareFaces = _lib
          .lookup<NativeFunction<CompareFacesC>>('compare_faces')
          .asFunction();

      _isInitialized = true;
    } catch (e) {
      print('Failed to initialize FaceRecognitionService: $e');
      rethrow;
    }
  }

  static Future<void> _copyAssetsToLocal() async {
    final assets = [
      'yolov5-face.onnx',
      'arcface_model.onnx'
    ];

    for (final asset in assets) {
      final localPath = await _getLocalPath(asset);
      if (!File(localPath).existsSync()) {
        try {
          final data = await rootBundle.load('assets/$asset');
          await File(localPath).writeAsBytes(data.buffer.asUint8List());
          print('Successfully copied $asset to $localPath');
        } catch (e) {
          print('Failed to copy $asset: $e');
          rethrow;
        }
      } else {
        print('$asset already exists at $localPath');
      }
    }
  }

  static Future<String> _getLocalPath(String fileName) async {
    final directory = await getApplicationDocumentsDirectory();
    return '${directory.path}/$fileName';
  }

  static bool get isInitialized => _isInitialized;

  static Future<List<FaceData>> recognizeFaces(List<String> imagePaths) async {
    final List<FaceData> allFaces = [];
    for (final imagePath in imagePaths) {
      final faces = await _detectFacesInImage(imagePath);
      allFaces.addAll(faces);
    }
    return allFaces;
  }

  static Future<List<FaceData>> _detectFacesInImage(String imagePath) async {
    final maxFaces = 10;
    final facesPtr = calloc<Int32>(maxFaces * 4);
    final faceDataPtr = calloc<Pointer<Uint8>>(maxFaces);
    final faceDataSizesPtr = calloc<Int32>(maxFaces);
    final faceFeaturesPtr = calloc<Pointer<Float>>(maxFaces);

    final result = _detectFaces(
      imagePath.toNativeUtf8(),
      _yolov5ModelPath.toNativeUtf8(),
      _arcfaceModelPath.toNativeUtf8(),
      facesPtr,
      maxFaces,
      faceDataPtr,
      faceDataSizesPtr,
      faceFeaturesPtr,
    );

    final List<FaceData> detectedFaces = [];

    if (result > 0) {
      for (int i = 0; i < result; i++) {
        final faceImage = faceDataPtr[i].asTypedList(faceDataSizesPtr[i]);
        final features = faceFeaturesPtr[i].asTypedList(128); // 假设特征向量长度为128

        detectedFaces.add(FaceData(
          faceImage: Uint8List.fromList(faceImage),
          features: Float32List.fromList(features),
        ));

        calloc.free(faceDataPtr[i]);
        calloc.free(faceFeaturesPtr[i]);
      }
    }

    calloc.free(facesPtr);
    calloc.free(faceDataPtr);
    calloc.free(faceDataSizesPtr);
    calloc.free(faceFeaturesPtr);

    return detectedFaces;
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
}

class FaceData {
  final Uint8List faceImage;
  final Float32List features;

  FaceData({required this.faceImage, required this.features});
}
