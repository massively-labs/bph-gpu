require "gnuplot"

class Shocktube
  def self.draw
    f = open("dat/shocktube.dat")
    content = f.read.split("\n")

    size = content.size
    step = 1.0 / size
    x = (0..size).map { |i| i*step + 0.5*step }
    y = content

    Gnuplot.open do |gp|
    Gnuplot::Plot.new( gp ) do |plot|
      plot.terminal "jpeg"
      plot.output "plot/shocktube.jpeg"
      plot.xlabel "x"
      plot.ylabel "density" 
      
      plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
        ds.with = "lines"
        ds.title = "shocktube"
      end
    end
    end
  end
end