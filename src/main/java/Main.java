import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class Main {

    private static Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws Exception {
        int seed = 123; // 난수 생성에 필요한 초기 시드값을 지정하기 위한 변수 (이 값이 바뀌면 SGD 알고리즘의 결과로 다른 local minimum value가 도출될 수 있음)
        double learningRate = 0.01; // Stochastic Gradient Descent 알고리즘에서 한번 미분할 때마다 좌우 방향으로 이동하는 최소 거리를 지정하기 위한 변수
        int batchSize = 50; // 한번에 처리하는 단위 데이터의 수를 지정하기 위한 변수 (여기서는 한번에 50개 데이터를 묶어서 처리하는 것으로 지정함)
        int nEpochs = 30; // 학습용 데이터를 전부 사용해서 학습시키는 반복 횟수를 지정하기 위한 변수 (여기서는 학습용 데이터 전부를 총 30번씩 사용하기로 지정함)

        // 입출력 레이어의 노드 수 및 히든 레이어의 노드 수를 지정하기 위한 변수들
        int numInputs = 2; // 입력 레이어에는 2개 노드
        int numOutputs = 2; // 출력 레이어에는 2개 노드
        int numHiddenNodes = 20; // 히든 레이어에는 20개 노드
        // 이 예제에서는 레이어는 입력, 히든, 출력 총 3개 레이어가 존재한다

        // 학습용, 평가용 데이터 파일경로 저장 (final 키워드는 java에서 상수를 의미함)
        final String filenameTrain  = new ClassPathResource("/classification/linear_data_train.csv").getFile().getPath();
        final String filenameTest  = new ClassPathResource("/classification/linear_data_eval.csv").getFile().getPath();

        // 학습용 데이터 로드
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize,0,2);

        // 평가용 데이터 로드
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize,0,2);

        // 딥 뉴럴넷 정의 (DL4J의 핵심이다)
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) // 초기 시드값 지정
                .iterations(1) // 학습 횟수 지정 ... 보니까 input 레이어는 따로 추가해주지 않는거 같고 그냥 seed 값만 지정해주면 되는거 같다!
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // 학습 알고리즘 지정
                .learningRate(learningRate) // SGD 의 learning rate 파라미터 지정
                .updater(Updater.NESTEROVS).momentum(0.9) // SGD 의 최적화 파라미터로 얼마나 local MIN 값을 빨리 찾게할지를 지정
                .list() // ? "리스트튼 신경망의 레이어 수를 지정합니다. 이 기능은 configuration을 n번 복제하고 레이어별로 구성할 수 있게 해줍니다 (?)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes) // 0번, 2개를 입력받아 20개를 내보낸다 (이 자체가 히든 레이어)
                        .weightInit(WeightInit.XAVIER) // xavier 초기화 지정
                        .activation(Activation.RELU) // RELU activation 펑션 지정
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // 1번, cost (loss) 펑션 지정
                        .weightInit(WeightInit.XAVIER) // xavier 초기화 지정
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER) // soft max activation 펑션 지정, soft max의 초기화는 xavier로 초기화
                        .nIn(numHiddenNodes).nOut(numOutputs).build()) // 20개를 받아서 2개를 내보낸다 (이 자체가 output 레이어)
                .pretrain(false).backprop(true).build(); // preTraining은 하지 않고 backProp만 한다

        MultiLayerNetwork model = new MultiLayerNetwork(conf); // 실제 멀티 레이어 딥 뉴럴 넷 모델을 만든다 (위에서 지정한 conf 지정)
        model.init(); // 초기화
        model.setListeners(new ScoreIterationListener(10));  // Print score every 10 parameter updates ?
        // 파라미터가 10번 바뀔 때마다 결과 값을 출력한다 ?

        for ( int n = 0; n < nEpochs; n++) { // 지정된 Epochs 수만큼 학습용 데이터를 사용해서 fitting 한다
            model.fit( trainIter ); // 학습용 데이터 지정
        }

        // 학습이 끝나면 모델을 평가 중이라고 출력해서 알려준다
        System.out.println("Evaluate model....");

        // 평가 모델 생성
        Evaluation eval = new Evaluation(numOutputs);

        // 평가 모델의 파라미터 지정
        while(testIter.hasNext()){
            DataSet t = testIter.next(); // 평가용 데이터 지정
            INDArray features = t.getFeatureMatrix(); // 피쳐 (문제)를 얻는다
            INDArray lables = t.getLabels(); // 레이블 (정답)을 얻는다
            INDArray predicted = model.output(features,false); // 예측값 (답)을 얻는다
            eval.eval(lables, predicted); // 실제 정답과, 예측값을 비교해서 평가한다
        }

        // 통계적으로 해당 모델을 평가한 내용을 출력한다
        System.out.println(eval.stats());

        //------------------------------------------------------------------------------------
        // Training is complete. Code that follows is for plotting the data & predictions only
        // 여기서부터는 학습 결과를 2D 그래프로 출력하는 과정의 코드이다

        // 2D 그래프의 x, y값 범위 지정
        double xMin = 0;
        double xMax = 1.0;
        double yMin = -0.2;
        double yMax = 0.8;

        // 총 100개의 포인트
        int nPointsPerAxis = 100;
        double[][] evalPoints = new double[nPointsPerAxis * nPointsPerAxis][2]; // x, y라서 2인건 알겠는데
        // 왜 100 * 100 = 10000 이 되어야 하는지 잘 모르겠다
        int count = 0; // 카운트를 세기 위한 변수

        // 모든 x, y 값에 대해
        for( int i=0; i<nPointsPerAxis; i++ ){
            for( int j=0; j<nPointsPerAxis; j++ )
            {
                // 뭔가를 수행한다 normalization 같아 보인다 ... ?
                double x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin;
                double y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin;

                evalPoints[count][0] = x;
                evalPoints[count][1] = y;

                count++;
            }
        }

        INDArray allXYPoints = Nd4j.create(evalPoints); // 점들을 만든다 ?
        INDArray predictionsAtXYPoints = model.output(allXYPoints); // 점들을 예측한다 ?

        // 모든 학습용 데이터를 단일 배열로 가져와서 플로팅 해본다
        rr.initialize(new FileSplit(new ClassPathResource("/classification/linear_data_train.csv").getFile())); // 학습용 데이터 파일 경로 저장
        rr.reset(); // 초기화
        int nTrainPoints = 1000; // 트레이닝 포인트 수 지정
        trainIter = new RecordReaderDataSetIterator(rr, nTrainPoints,0,2); // 학습용 데이터 rr은 총 1000개 이고 0아니면 1값을 갖는다
        DataSet ds = trainIter.next(); // 데이터셋에 trainIter 추가 ...?
        PlotUtil.plotTrainingData(ds.getFeatures(), ds.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis); // 플로팅해본다

        // 평가용 데이터를 얻고, 딥 뉴럴 넷을 통해 예측값을 생성하고 해당 예측값을 플로팅 해본다
        rrTest.initialize(new FileSplit(new ClassPathResource("/classification/linear_data_eval.csv").getFile())); // 평가용 데이터 파일 경로 저장
        rrTest.reset(); // 초기화
        int nTestPoints = 500; // 평가용 포인트 수 지정 ... 위 파일에는 데이터가 총 200개인데 왜 여기서는 500으로 지정되어 있을까? 의문이다 ...
        testIter = new RecordReaderDataSetIterator(rrTest, nTestPoints,0,2); // 평가용 데이터 rrTest는 총 200개인데
        ds = testIter.next(); // 데이터셋에 testIter 추가 ...?

        INDArray testPredicted = model.output(ds.getFeatures()); // 딥 뉴럴 넷 모델을 통해 예측값을 생성해본다
        PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis); // 예측한 내용을 플로팅해본다

        System.out.println("****************Example finished********************");
    }
}