import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'face_recognition_service.dart';

class PhotoState extends ChangeNotifier {
  List<XFile> _imageFileList = [];
  Map<String, List<Uint8List>> _detectedFaces = {};
  bool _isProcessing = false;

  List<XFile> get imageFileList => _imageFileList;
  bool get isProcessing => _isProcessing;

  bool faceDetected(int index) {
    return _detectedFaces.containsKey(_imageFileList[index].path) &&
        _detectedFaces[_imageFileList[index].path]!.isNotEmpty;
  }

  Future<void> addImages(List<XFile> newImages) async {
    _imageFileList.addAll(newImages);
    notifyListeners();
    
    if (!FaceRecognitionService.isInitialized) {
      print('FaceRecognitionService is not initialized. Attempting to initialize...');
      try {
        await FaceRecognitionService.initialize();
      } catch (e) {
        print('Failed to initialize FaceRecognitionService: $e');
        return;
      }
    }

    if (!FaceRecognitionService.isInitialized) {
      print('FaceRecognitionService is still not initialized. Skipping face recognition.');
      return;
    }

    _isProcessing = true;
    notifyListeners();

    try {
      List<Uint8List> faceImages = await FaceRecognitionService.recognizeFaces(newImages.map((file) => file.path).toList());
      for (int i = 0; i < newImages.length; i++) {
        _detectedFaces[newImages[i].path] = faceImages;
      }
    } catch (e) {
      print('Error during face recognition: $e');
    } finally {
      _isProcessing = false;
      notifyListeners();
    }
  }

  void clearImages() {
    _imageFileList.clear();
    _detectedFaces.clear();
    notifyListeners();
  }
}