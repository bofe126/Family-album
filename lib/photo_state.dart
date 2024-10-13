import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'face_recognition_service.dart';

class UniqueFace {
  final FaceData faceData;
  final List<XFile> images;
  String? label;

  UniqueFace(this.faceData, this.images);

  void addImage(XFile image) {
    if (!images.contains(image)) {
      images.add(image);
    }
  }
}

class PhotoState extends ChangeNotifier {
  List<XFile> _imageFileList = [];
  Map<String, List<FaceData>> _detectedFaces = {};
  List<UniqueFace> _uniqueFaces = [];
  UniqueFace? _selectedFace;
  double _similarityThreshold = 0.7; // 降低阈值
  String _searchQuery = '';
  double _scoreThreshold = 0.6;

  List<XFile> get imageFileList => _imageFileList;
  List<UniqueFace> get uniqueFaces => _uniqueFaces;
  UniqueFace? get selectedFace => _selectedFace;
  double get similarityThreshold => _similarityThreshold;
  String get searchQuery => _searchQuery;
  double get scoreThreshold => _scoreThreshold;

  List<XFile> get filteredImages {
    print("正在过滤图片。总数: ${_imageFileList.length}, 选中的人脸: ${_selectedFace?.label ?? '无'}, 搜索查询: $_searchQuery");
    if (_selectedFace != null) {
      print("按选中的人脸过滤: ${_selectedFace!.label ?? 'Unlabeled'}");
      return _selectedFace!.images;
    } else if (_searchQuery.isNotEmpty) {
      print("按搜索查询过滤: $_searchQuery");
      return _imageFileList
          .where((image) =>
              _detectedFaces[image.path]?.any((face) => _uniqueFaces.any(
                  (uniqueFace) =>
                      uniqueFace.label
                          ?.toLowerCase()
                          .contains(_searchQuery.toLowerCase()) ??
                      false)) ??
              false)
          .toList();
    } else {
      print("没有应用过滤，返回所有图片");
      return _imageFileList;
    }
  }

  bool faceDetected(String imagePath) {
    return _detectedFaces.containsKey(imagePath) &&
        _detectedFaces[imagePath]!.isNotEmpty;
  }

  void addImages(List<XFile> newImages) async {
    print("正在添加 ${newImages.length} 张新图片");
    _imageFileList.addAll(newImages);
    await _detectFaces(newImages);
    _clusterFaces();
    notifyListeners();
    print("添加后的总图片数: ${_imageFileList.length}");
  }

  Future<void> _detectFaces(List<XFile> images) async {
    for (var image in images) {
      List<FaceData> faces = await FaceRecognitionService.detectFaces(
        image.path,
        scoreThreshold: _scoreThreshold
      );
      _detectedFaces[image.path] = faces;
      for (var face in faces) {
        bool added = false;
        for (var uniqueFace in _uniqueFaces) {
          double similarity = FaceRecognitionService.compareFaces(
              face.features, uniqueFace.faceData.features);
          if (similarity > _similarityThreshold) {
            uniqueFace.addImage(image);
            added = true;
            break;
          }
        }
        if (!added) {
          _uniqueFaces.add(UniqueFace(face, [image]));
        }
      }
    }
    _clusterFaces();
  }

  void _clusterFaces() {
    List<UniqueFace> newUniqueFaces = [];
    for (var face in _uniqueFaces) {
      bool added = false;
      for (var uniqueFace in newUniqueFaces) {
        double similarity = FaceRecognitionService.compareFaces(
            face.faceData.features, uniqueFace.faceData.features);
        if (similarity > _similarityThreshold) {
          uniqueFace.images.addAll(face.images);
          added = true;
          break;
        }
      }
      if (!added) {
        newUniqueFaces.add(face);
      }
    }
    _uniqueFaces = newUniqueFaces;
  }

  void selectFace(UniqueFace face) {
    _selectedFace = face;
    _searchQuery = '';
    notifyListeners();
  }

  void setSimilarityThreshold(double value) {
    _similarityThreshold = value;
    _clusterFaces();
    notifyListeners();
  }

  void labelFace(UniqueFace face, String label) {
    face.label = label;
    notifyListeners();
  }

  void setSearchQuery(String query) {
    _searchQuery = query;
    _selectedFace = null;
    notifyListeners();
  }

  String visualizeFeatures(Float32List features) {
    return features.sublist(0, 10).map((f) => f.toStringAsFixed(2)).join(', ') +
        '...';
  }

  Future<void> detectFaces() async {
    if (!FaceRecognitionService.isInitialized) {
      await FaceRecognitionService.initialize();
    }

    _imageFileList.map((photo) => photo.path).toList();
    
    // 处理检测到的人脸数据
    // ...

    notifyListeners();
  }

  Future<void> processPhotos() async {
    // 实现照片处理逻辑
    // 1. 遍历所有未处理的照片
    // 2. 对每张照片进行人脸检测
    // 3. 提取人脸特征
    // 4. 更新照片和人脸数据
    // 5. 通知监听器
  }

  void setScoreThreshold(double value) {
    _scoreThreshold = value;
    notifyListeners();
  }
}
