import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.LineReader;

public class BinaryFilesToHadoopSequenceFile {

	private static Log logger = LogFactory
			.getLog(BinaryFilesToHadoopSequenceFile.class);

	public static class ImageToSequenceFileMapper extends
			Mapper<Object, Text, Text, BytesWritable> {

		private static int reducerNumber = 0;

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {

			// logger.info("map method called.. with value: " +
			// value.toString());
			// logger.info(context.getConfiguration().get("NLINESTOPROCESS"));
			String lines[] = value.toString().split("\n");
			int numLines = lines.length;
			logger.info("received " + numLines + " files\n");
			int numReducers = Integer.parseInt(context.getConfiguration().get(
					"NUMREDUCERS"));

			Configuration conf = new Configuration();
			for (int i = 0; i < numLines; i++) {
				String uri = lines[i].toString();
				FileSystem fs = FileSystem.get(URI.create(uri), conf);
				FSDataInputStream in = null;

				try {
					in = fs.open(new Path(uri));
					// b = ImageIO.read(in);
					// int height = b.getHeight();
					// int width = b.getWidth();
					// String s = "@@" + height + "X" + width + "@@";
					in.seek(0);
					BytesWritable v = new BytesWritable(
							org.apache.commons.io.IOUtils.toByteArray(in));
					// while( in.read(buffer, 0, buffer.length) >= 0 ) {
					// bout.write(buffer);
					// }
					String s = lines[i].toString() + "_r_"
							+ Integer.toString(reducerNumber % numReducers);
					value.set(s);
					context.write(value, v);
					logger.info("emiiting key - " + s);
					reducerNumber++;
				} finally {
					IOUtils.closeStream(in);
				}
			}
		}
	}

	public static class SequenceFilePartitioner extends
			Partitioner<Text, BytesWritable> {

		@Override
		public int getPartition(Text key, BytesWritable value,
				int numReduceTasks) {

			String a[] = key.toString().split("_r_");
			logger.info("sending " + key.toString() + " to " + a[a.length - 1]);
			return Integer.parseInt(a[a.length - 1]);
		}
	}

	public static class ImageToSequenceFileReducer extends
			Reducer<Text, BytesWritable, Text, BytesWritable> {

		private MultipleOutputs<Text, BytesWritable> mos;

		protected void setup(Context context) throws IOException,
				InterruptedException {
			mos = new MultipleOutputs<Text, BytesWritable>(context);
		}

		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			mos.close();
		}

		@Override
		public void reduce(Text key, Iterable<BytesWritable> values,
				Context context) throws IOException, InterruptedException {
			String filename = "n" + getFilenameFromKey(key);
			/**
			 * To Do: remove "_r_" from key.
			 **/
			// filename =
			// context.getConfiguration().get("BASE_OUTPUT_FILE_NAME")
			// + "n" + filename;
			for (BytesWritable value : values) {
				// mos.write(key, value, filename);
				mos.write(filename, key, value);
			}
		}

		private String getFilenameFromKey(Text key) {
			String a[] = key.toString().split("_r_");
			return a[a.length - 1];
		}
	}

	public static class NLinesInputFormat extends TextInputFormat {
		public RecordReader<LongWritable, Text> createRecordReader(
				InputSplit split, TaskAttemptContext context) {
			logger.info("new recordreader() being called\n");
			return new NLinesRecordReader();
		}
	}

	public static class NLinesRecordReader extends
			RecordReader<LongWritable, Text> {
		private int NLINESTOPROCESS = 3;
		private LineReader in;
		private LongWritable key;
		private Text value = new Text();
		private long start = 0;
		private long end = 0;
		private long pos = 0;
		private int maxLineLength;

		// private Log logger = LogFactory
		// .getLog(NLinesRecordReader.class);

		@Override
		public void close() throws IOException {
			if (in != null) {
				in.close();
			}
		}

		@Override
		public LongWritable getCurrentKey() throws IOException,
				InterruptedException {
			return key;
		}

		@Override
		public Text getCurrentValue() throws IOException, InterruptedException {
			return value;
		}

		@Override
		public float getProgress() throws IOException, InterruptedException {
			if (start == end) {
				return 0.0f;
			} else {
				return Math.min(1.0f, (pos - start) / (float) (end - start));
			}
		}

		@Override
		public void initialize(InputSplit genericSplit,
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			FileSplit split = (FileSplit) genericSplit;
			final Path file = split.getPath();
			Configuration conf = context.getConfiguration();

			// get the parameter from the configuration
			NLINESTOPROCESS = Integer.parseInt(conf.get("NLINESTOPROCESS"));
			logger.info("initialize() called.., nltp = "
					+ Integer.toString(NLINESTOPROCESS));

			this.maxLineLength = conf.getInt(
					"mapreduce.linerecordreader.maxlength", Integer.MAX_VALUE);
			FileSystem fs = file.getFileSystem(conf);
			start = split.getStart();
			end = start + split.getLength();
			boolean skipFirstLine = false;
			FSDataInputStream filein = fs.open(split.getPath());

			if (start != 0) {
				skipFirstLine = true;
				--start;
				filein.seek(start);
			}
			in = new LineReader(filein, conf);
			if (skipFirstLine) {
				start += in.readLine(new Text(), 0,
						(int) Math.min((long) Integer.MAX_VALUE, end - start));
			}
			this.pos = start;
		}

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			logger.info("nextKeyValue() called\n");
			if (key == null) {
				key = new LongWritable();
			}
			key.set(pos);
			if (value == null) {
				value = new Text();
			}
			value.clear();
			final Text endline = new Text("\n");
			int newSize = 0;
			for (int i = 0; i < NLINESTOPROCESS; i++) {
				Text v = new Text();
				while (pos < end) {
					newSize = in.readLine(
							v,
							maxLineLength,
							Math.max((int) Math.min(Integer.MAX_VALUE, end
									- pos), maxLineLength));
					value.append(v.getBytes(), 0, v.getLength());
					value.append(endline.getBytes(), 0, endline.getLength());
					if (newSize == 0) {
						break;
					}
					pos += newSize;
					if (newSize < maxLineLength) {
						break;
					}
				}
			}
			if (newSize == 0) {
				key = null;
				value = null;
				return false;
			} else {
				return true;
			}
		}
	}

	public static void main(String[] args) throws Exception {

		if (args.length < 4) {
			System.err
					.println("Usage: BinaryFilesToHadoopSequenceFile <in Path for url file> "
							+ "<approximate number of image files> "
							+ "<number of sequence files to output>"
							+ "<ABSOLUTE path to hdfs locaiton where the output folder "
							+ "will automatically be created>");
			System.exit(2);
		}

		int numOutputFiles = Integer.parseInt(args[args.length - 2]);
		int approxInputFiles = Integer.parseInt(args[args.length - 3]);

		if (numOutputFiles < 1) {
			// someone is screwing around
			numOutputFiles = 1;
		}
		String absPath = args[args.length - 1];
		if (absPath.charAt(absPath.length() - 1) != '/') {
			absPath += "/";
		}

		DateFormat dateFormat = new SimpleDateFormat("ddMMyyyyHHmmss");
		Date date = new Date();
		String outputfolder = absPath + "IToSeq" + dateFormat.format(date);

		// remember to set conf value before creating job instance
		Configuration conf = new Configuration();
		conf.set("BASE_OUTPUT_FILE_NAME", outputfolder);
		conf.set("NLINESTOPROCESS", Integer
				.toString(((int) (approxInputFiles / numOutputFiles) + 1)));
		conf.set("NUMREDUCERS", Integer.toString((int) numOutputFiles));

		@SuppressWarnings("deprecation")
		Job job = new Job(conf);
		job.setJarByClass(BinaryFilesToHadoopSequenceFile.class);
		job.setMapperClass(ImageToSequenceFileMapper.class);
		job.setReducerClass(ImageToSequenceFileReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(BytesWritable.class);
		job.setInputFormatClass(NLinesInputFormat.class);
		LazyOutputFormat.setOutputFormatClass(job,
				SequenceFileOutputFormat.class);

		for (int i = 0; i < args.length - 3; i++) {
			MultipleInputs.addInputPath(job, new Path(args[i]),
					NLinesInputFormat.class);
		}
		job.setPartitionerClass(SequenceFilePartitioner.class);
		job.setNumReduceTasks(numOutputFiles);

		for (int i = 0; i < numOutputFiles; i++) {
			MultipleOutputs.addNamedOutput(job, "n" + Integer.toString(i),
					SequenceFileOutputFormat.class, Text.class,
					BytesWritable.class);
		}

		FileOutputFormat.setOutputPath(job, new Path(outputfolder));

		Path f = new Path(absPath + "IToSeq.outputlocation");
		FileSystem fs = FileSystem.get(conf);
		if (fs.exists(f)) {
			// File already exists.
			// Delete the file before proceeding.
			logger.info("Deleting existing file");
			fs.delete(f, true);
		}
		FSDataOutputStream os = fs.create(f);
		BufferedWriter br = new BufferedWriter(new OutputStreamWriter(os,
				"UTF-8"));
		br.write(outputfolder);
		br.close();
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}